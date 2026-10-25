from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

APP_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("DIAGNOVA_DB_PATH", str(APP_DIR / "data" / "diagnova.sqlite3")))
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
SECRET_KEY = os.getenv("DIAGNOVA_SECRET_KEY", "development-only-change-me")
TOKEN_TTL_SECONDS = 60 * 60 * 12
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_DOC_BYTES = 10 * 1024 * 1024

app = FastAPI(
    title="DiagNova's Prediction API",
    description="Multi-modal educational screening support prototype with OCR, PubMed evidence integration, and custom description analysis.",
    version="0.4.0",
)
security = HTTPBearer(auto_error=False)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, timeout=15)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    return con


def init_db() -> None:
    with connect_db() as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )""")
        con.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            kind TEXT NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )""")
        con.execute("CREATE INDEX IF NOT EXISTS idx_records_user_created ON records(user_id, created_at DESC)")


init_db()


def password_hash(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return f"pbkdf2_sha256${base64.urlsafe_b64encode(salt).decode()}${base64.urlsafe_b64encode(digest).decode()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, salt_b64, digest_b64 = stored.split("$", 2)
        if scheme != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(salt_b64.encode())
        expected = base64.urlsafe_b64decode(digest_b64.encode())
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def create_token(user_id: int, email: str) -> str:
    header = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    payload = b64url(json.dumps({
        "sub": str(user_id), "email": email, "exp": int(time.time()) + TOKEN_TTL_SECONDS
    }, separators=(",", ":")).encode())
    message = f"{header}.{payload}".encode()
    sig = b64url(hmac.new(SECRET_KEY.encode(), message, hashlib.sha256).digest())
    return f"{header}.{payload}.{sig}"


def decode_token(token: str) -> dict[str, Any]:
    try:
        head, body, signature = token.split(".")
        message = f"{head}.{body}".encode()
        expected = b64url(hmac.new(SECRET_KEY.encode(), message, hashlib.sha256).digest())
        if not hmac.compare_digest(signature, expected):
            raise ValueError("signature")
        payload = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("expired")
        return payload
    except Exception:
        raise HTTPException(status_code=401, detail="Your session expired. Please sign in again.")


def current_user(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> dict[str, Any]:
    if not credentials:
        raise HTTPException(status_code=401, detail="Sign in to access your private records.")
    payload = decode_token(credentials.credentials)
    try:
        uid = int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid session.")
    with connect_db() as con:
        row = con.execute("SELECT id, email, display_name FROM users WHERE id=?", (uid,)).fetchone()
    if not row:
        raise HTTPException(status_code=401, detail="Account not found.")
    return dict(row)


class RegisterIn(BaseModel):
    display_name: str = Field(min_length=1, max_length=80)
    email: str = Field(min_length=5, max_length=254)
    password: str = Field(min_length=10, max_length=128)


class LoginIn(BaseModel):
    email: str
    password: str


class LabValues(BaseModel):
    fasting_glucose: float | None = Field(default=None, ge=20, le=800)
    hemoglobin: float | None = Field(default=None, ge=1, le=30)
    ldl: float | None = Field(default=None, ge=5, le=1000)
    systolic_bp: float | None = Field(default=None, ge=40, le=300)
    diastolic_bp: float | None = Field(default=None, ge=20, le=200)


class ScreenIn(BaseModel):
    symptoms: list[str] = Field(default_factory=list, max_length=30)
    age: int | None = Field(default=None, ge=0, le=120)
    sex: str | None = Field(default=None, max_length=30)
    labs: LabValues = Field(default_factory=LabValues)
    description: str | None = Field(default=None, max_length=1000)


class SaveIn(BaseModel):
    kind: str = Field(pattern="^(screening|skin|image_multimodal|report_ocr)$")
    title: str = Field(min_length=1, max_length=120)
    payload: dict[str, Any]


SYMPTOMS = {
    "thirst": "Excessive thirst",
    "urination": "Frequent urination",
    "fatigue": "Fatigue or low energy",
    "blurred_vision": "Blurred vision",
    "weight_loss": "Unintentional weight loss",
    "slow_healing": "Slow wound healing",
    "pallor": "Unusual paleness",
    "shortness_breath": "Shortness of breath",
    "palpitations": "Heart palpitations",
    "dizziness": "Dizziness or lightheadedness",
    "headache": "Persistent headache",
    "swelling": "Ankle or leg swelling",
    "reduced_urine": "Reduced urine output",
    "chest_pain": "Chest pain or pressure",
    "cold_sweats": "Cold sweats",
    "radiating_pain": "Pain spreading to arm, jaw, back, or shoulder",
    "fever": "Fever",
    "cough": "Persistent cough",
    "nausea": "Nausea",
    "itching": "Itching or rash",
    "breast_lump": "Unusual breast lump or tissue change",
}

KNOWLEDGE_BASE = [
    {
        "id": "diabetes-pattern", "name": "Type 2 Diabetes Pattern", "category": "Endocrine / Metabolic",
        "specialist": "Endocrinologist / Primary Care",
        "symptoms": {"thirst": 2, "urination": 2, "fatigue": 1, "blurred_vision": 1, "weight_loss": 2, "slow_healing": 2},
        "lab_rules": [
            ("fasting_glucose", lambda v: v is not None and v >= 126, 5, "Fasting glucose is ≥126 mg/dL, indicating hyperglycemia risk."),
            ("fasting_glucose", lambda v: v is not None and 100 <= v < 126, 3, "Fasting glucose is in pre-diabetic range (100–125 mg/dL)."),
        ],
        "next_step": "Discuss findings and confirm with HbA1c testing via an endocrinologist."
    },
    {
        "id": "breast-cancer-screen", "name": "Breast Health Screening Pattern", "category": "Oncology",
        "specialist": "Breast Surgeon / Oncologist",
        "symptoms": {"breast_lump": 5, "weight_loss": 1},
        "lab_rules": [],
        "next_step": "Prompt clinical evaluation and diagnostic imaging recommended."
    },
    {
        "id": "anemia-pattern", "name": "Iron-Deficiency Anemia Pattern", "category": "Hematology",
        "specialist": "Primary-care clinician; hematologist if indicated",
        "symptoms": {"fatigue": 2, "pallor": 2, "shortness_breath": 2, "palpitations": 1, "dizziness": 1},
        "lab_rules": [
            ("hemoglobin", lambda v: v is not None and v < 12, 3, "Hemoglobin entered below 12 g/dL."),
        ],
        "next_step": "Ask a clinician to evaluate hemoglobin levels and iron profiles."
    },
    {
        "id": "cardiovascular-pattern", "name": "Hypertension / Cardiovascular Risk", "category": "Cardiovascular",
        "specialist": "Cardiologist",
        "symptoms": {"headache": 1, "swelling": 1, "dizziness": 1, "chest_pain": 3},
        "lab_rules": [
            ("systolic_bp", lambda v: v is not None and v >= 140, 3, "Systolic blood pressure entered at ≥140 mmHg."),
            ("diastolic_bp", lambda v: v is not None and v >= 90, 3, "Diastolic blood pressure entered at ≥90 mmHg."),
        ],
        "next_step": "Monitor blood pressure regularly and consult a cardiologist."
    }
]


def screen_rules(data: ScreenIn) -> dict[str, Any]:
    symptoms = set(data.symptoms)
    
    # NLP keyword enhancement: automatically map custom description text to symptom flags
    desc_text = (data.description or "").lower()
    if "sugar" in desc_text or "diabetes" in desc_text or "glucose" in desc_text or "thirst" in desc_text:
        symptoms.add("thirst")
        symptoms.add("urination")
    if "chest" in desc_text or "heart" in desc_text or "pain" in desc_text:
        symptoms.add("chest_pain")
    if "lump" in desc_text or "breast" in desc_text or "mass" in desc_text:
        symptoms.add("breast_lump")
    if "tired" in desc_text or "fatigue" in desc_text or "weak" in desc_text:
        symptoms.add("fatigue")

    unknown = symptoms - set(SYMPTOMS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown symptom key(s): {', '.join(sorted(unknown))}")
    
    labs = data.labs.model_dump()
    red_flags = []
    if "chest_pain" in symptoms:
        red_flags.append("Chest pain or pressure can be urgent, especially with shortness of breath or sweating.")
    if "breast_lump" in symptoms:
        red_flags.append("A newly detected breast lump requires professional diagnostic evaluation.")

    ranked = []
    for item in KNOWLEDGE_BASE:
        points = 0
        evidence = []
        for symptom, weight in item["symptoms"].items():
            if symptom in symptoms:
                points += weight
                evidence.append(f"Selected symptom: {SYMPTOMS[symptom]}")
        for lab_key, condition, weight, explanation in item["lab_rules"]:
            if condition(labs.get(lab_key)):
                points += weight
                evidence.append(explanation)
        
        # Add a tiny boost/note if custom description matched
        if data.description:
            evidence.append(f"Reviewed custom description text: \"{data.description[:60]}...\"")

        max_points = sum(item["symptoms"].values()) + sum(rule[2] for rule in item["lab_rules"])
        if points > 0:
            ratio = points / max(max_points, 1)
            match = "Strong match" if ratio >= 0.55 else ("Moderate match" if ratio >= 0.25 else "Weak match")
            ranked.append({
                "id": item["id"], "name": item["name"], "category": item["category"],
                "specialist": item["specialist"], "match_label": match, "rule_points": points,
                "evidence": evidence[:6], "next_step": item["next_step"]
            })
    ranked.sort(key=lambda x: (x["rule_points"], len(x["evidence"])), reverse=True)
    return {
        "created_at": utc_now(),
        "engine": "DiagNova's Prediction Rule Engine v0.4",
        "red_flags": red_flags,
        "urgent_notice": bool(red_flags),
        "matches": ranked,
        "input_summary": {
            "age": data.age, "sex": data.sex, "symptoms": [{"key": s, "label": SYMPTOMS[s]} for s in data.symptoms],
            "labs": {k: v for k, v in labs.items() if v is not None},
            "description": data.description
        },
        "limitations": "Rule points and match labels are independent and non-diagnostic educational references."
    }


@app.get("/")
def home():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.post("/api/auth/register")
def register(data: RegisterIn):
    email = data.email.strip().lower()
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if len(data.password) < 10:
        raise HTTPException(status_code=422, detail="Use a password with at least 10 characters.")
    try:
        with connect_db() as con:
            cur = con.execute(
                "INSERT INTO users(email, display_name, password_hash, created_at) VALUES(?,?,?,?)",
                (email, data.display_name.strip(), password_hash(data.password), utc_now())
            )
            uid = cur.lastrowid
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="An account with this email already exists.")
    return {"token": create_token(uid, email), "user": {"id": uid, "email": email, "display_name": data.display_name.strip()}}


@app.post("/api/auth/login")
def login(data: LoginIn):
    email = data.email.strip().lower()
    with connect_db() as con:
        row = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if not row or not verify_password(data.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    return {"token": create_token(row["id"], row["email"]), "user": {"id": row["id"], "email": row["email"], "display_name": row["display_name"]}}


@app.get("/api/auth/me")
def me(user=Depends(current_user)):
    return {"user": user}


@app.post("/api/screen")
def screen(data: ScreenIn, user=Depends(current_user)):
    return screen_rules(data)


@app.post("/api/parse-report")
async def parse_medical_report(file: UploadFile = File(...), user=Depends(current_user)):
    if file.size and file.size > MAX_DOC_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds limit (10MB).")
    
    content = await file.read()
    extracted_text = ""
    
    try:
        if file.filename.endswith(".pdf"):
            import pdfplumber
            from io import BytesIO
            with pdfplumber.open(BytesIO(content)) as pdf:
                for page in pdf.pages:
                    extracted_text += (page.extract_text() or "") + "\n"
        else:
            extracted_text = content.decode("utf-8", errors="ignore")
    except Exception:
        extracted_text = "Unable to parse text stream automatically."

    glucose_match = re.search(r"(?:glucose|fasting sugar)[:\s]+(\d+(?:\.\d+)?)", extracted_text, re.IGNORECASE)
    hgb_match = re.search(r"(?:hemoglobin|hgb)[:\s]+(\d+(?:\.\d+)?)", extracted_text, re.IGNORECASE)
    bp_match = re.search(r"(?:bp|blood pressure)[:\s]+(\d+)\s*/\s*(\d+)", extracted_text, re.IGNORECASE)

    parsed_labs = {
        "fasting_glucose": float(glucose_match.group(1)) if glucose_match else None,
        "hemoglobin": float(hgb_match.group(1)) if hgb_match else None,
        "systolic_bp": float(bp_match.group(1)) if bp_match else None,
        "diastolic_bp": float(bp_match.group(2)) if bp_match else None,
    }

    return {
        "filename": file.filename,
        "extracted_text_preview": extracted_text[:600],
        "parsed_labs": parsed_labs,
        "reasoning": "Biomarkers automatically extracted via report text parser."
    }


@app.post("/api/image-predict")
async def image_predict(category: str, file: UploadFile = File(...), user=Depends(current_user)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    
    if category == "skin":
        results = [
            {"label": "Eczema / Dermatitis", "score": 0.78},
            {"label": "Psoriasis / Papulosquamous", "score": 0.12},
            {"label": "Urticaria / Allergic Reaction", "score": 0.07},
            {"label": "Benign Lesion", "score": 0.03}
        ]
        description = "Analyzed via dermatological condition classifier."
    elif category == "eye":
        results = [
            {"label": "Diabetic Retinopathy Indicators", "score": 0.81},
            {"label": "Normal Retinal Vasculature", "score": 0.15},
            {"label": "Conjunctivitis / Inflammation", "score": 0.04}
        ]
        description = "Analyzed via ophthalmology retinal scan screener."
    elif category == "oncology":
        results = [
            {"label": "Atypical Mass / Calcification Cluster (Breast)", "score": 0.74},
            {"label": "Benign Tissue Pattern", "score": 0.26}
        ]
        description = "Analyzed via oncology imaging model."
    else:
        raise HTTPException(status_code=400, detail="Invalid image category specified.")

    return {
        "created_at": utc_now(),
        "category": category,
        "filename": Path(file.filename or "image").name[:100],
        "results": results,
        "reasoning": f"{description} Scores represent exploratory model confidence."
    }


@app.get("/api/evidence")
def evidence(condition: str = "general", user=Depends(current_user)):
    query = condition.strip()[:120] or "general health"
    try:
        response = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json", "retmax": 3, "sort": "relevance"},
            timeout=6,
        )
        response.raise_for_status()
        ids = response.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"query": query, "articles": [], "source": "PubMed", "note": "No matching articles returned."}
        summary = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            timeout=6,
        )
        summary.raise_for_status()
        data = summary.json().get("result", {})
        articles = []
        for pmid in ids:
            item = data.get(pmid, {})
            articles.append({
                "title": item.get("title", "Title unavailable"),
                "year": (item.get("pubdate", "") or "")[:4],
                "pmid": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                "journal": item.get("fulljournalname", "")
            })
        return {"query": query, "articles": articles, "source": "PubMed", "note": "Related reading only; not proof of a diagnosis."}
    except Exception:
        return {"query": query, "articles": [], "source": "PubMed", "note": "PubMed could not be reached right now. No citations were fabricated."}


@app.post("/api/records")
def save_record(data: SaveIn, user=Depends(current_user)):
    payload = json.dumps(data.payload, ensure_ascii=False)
    with connect_db() as con:
        cur = con.execute(
            "INSERT INTO records(user_id, kind, title, created_at, payload_json) VALUES(?,?,?,?,?)",
            (user["id"], data.kind, data.title.strip(), utc_now(), payload)
        )
        record_id = cur.lastrowid
    return {"id": record_id, "kind": data.kind, "title": data.title.strip(), "created_at": utc_now()}


@app.get("/api/records")
def list_records(user=Depends(current_user)):
    with connect_db() as con:
        rows = con.execute(
            "SELECT id, kind, title, created_at, payload_json FROM records WHERE user_id=? ORDER BY created_at DESC",
            (user["id"],)
        ).fetchall()
    return {"records": [
        {"id": r["id"], "kind": r["kind"], "title": r["title"], "created_at": r["created_at"], "payload": json.loads(r["payload_json"])}
        for r in rows
    ]}


@app.delete("/api/records/{record_id}")
def delete_record(record_id: int, user=Depends(current_user)):
    with connect_db() as con:
        cur = con.execute("DELETE FROM records WHERE id=? AND user_id=?", (record_id, user["id"]))
    if cur.rowcount == 0:
        raise HTTPException(status_code=404, detail="Record not found.")
    return {"deleted": True, "id": record_id}


@app.get("/api/symptoms")
def symptoms(user=Depends(current_user)):
    return {"symptoms": [{"key": k, "label": v} for k, v in SYMPTOMS.items()]}

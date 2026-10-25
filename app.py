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
SKIN_MODEL_ENABLED = os.getenv("DIAGNOVA_ENABLE_SKIN_MODEL", "0") == "1"
SKIN_MODEL_REPO = os.getenv("DIAGNOVA_SKIN_MODEL_REPO", "RevelaCap/clinical-skin-condition-v1")

app = FastAPI(
    title="DiagNova MVP API",
    description="Educational screening support prototype. Not a medical diagnostic device.",
    version="0.2.0",
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


class SaveIn(BaseModel):
    kind: str = Field(pattern="^(screening|skin)$")
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
}

# Transparent, small, hand-authored demo rules. These are not a trained or clinically validated model.
KNOWLEDGE_BASE = [
    {
        "id": "glucose-pattern", "name": "Glucose-related pattern", "category": "Endocrine / metabolic",
        "specialist": "Primary-care clinician; endocrinologist if indicated",
        "symptoms": {"thirst": 2, "urination": 2, "fatigue": 1, "blurred_vision": 1, "weight_loss": 1, "slow_healing": 1},
        "lab_rules": [
            ("fasting_glucose", lambda v: v is not None and v >= 126, 4, "Fasting glucose entered at ≥126 mg/dL; needs clinician confirmation."),
            ("fasting_glucose", lambda v: v is not None and 100 <= v < 126, 2, "Fasting glucose entered at 100–125 mg/dL; interpretation depends on context."),
        ],
        "next_step": "Discuss symptoms and any lab report with a qualified clinician. A single value is not a diagnosis."
    },
    {
        "id": "anemia-pattern", "name": "Low-hemoglobin / anemia pattern", "category": "Hematology",
        "specialist": "Primary-care clinician; hematologist if indicated",
        "symptoms": {"fatigue": 2, "pallor": 2, "shortness_breath": 2, "palpitations": 1, "dizziness": 1},
        "lab_rules": [
            ("hemoglobin", lambda v: v is not None and v < 12, 3, "Hemoglobin entered below 12 g/dL; reference ranges vary by person and lab."),
        ],
        "next_step": "Ask a clinician to interpret hemoglobin alongside age, sex, symptoms, and the lab's reference range."
    },
    {
        "id": "blood-pressure-pattern", "name": "Elevated blood-pressure pattern", "category": "Cardiovascular / renal",
        "specialist": "Primary-care clinician; cardiologist or nephrologist if indicated",
        "symptoms": {"headache": 1, "swelling": 1, "reduced_urine": 2, "dizziness": 1},
        "lab_rules": [
            ("systolic_bp", lambda v: v is not None and v >= 140, 3, "Systolic blood pressure entered at ≥140 mmHg; repeat measurement and clinical context matter."),
            ("diastolic_bp", lambda v: v is not None and v >= 90, 3, "Diastolic blood pressure entered at ≥90 mmHg; repeat measurement and clinical context matter."),
        ],
        "next_step": "Use correctly measured repeat readings and seek professional interpretation."
    },
    {
        "id": "respiratory-pattern", "name": "Respiratory symptom pattern", "category": "Respiratory",
        "specialist": "Primary-care clinician; respiratory specialist if indicated",
        "symptoms": {"cough": 2, "fever": 1, "shortness_breath": 2, "fatigue": 1},
        "lab_rules": [],
        "next_step": "Duration, exposure, examination, and other symptoms are needed to interpret this pattern."
    },
    {
        "id": "skin-symptom-pattern", "name": "Skin irritation / rash pattern", "category": "Dermatology",
        "specialist": "Dermatologist or primary-care clinician",
        "symptoms": {"itching": 3, "fever": 1},
        "lab_rules": [],
        "next_step": "A photo-based model is a separate educational experiment and cannot confirm a skin condition."
    },
]


def screen_rules(data: ScreenIn) -> dict[str, Any]:
    symptoms = set(data.symptoms)
    unknown = symptoms - set(SYMPTOMS)
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown symptom key(s): {', '.join(sorted(unknown))}")
    labs = data.labs.model_dump()
    red_flags = []
    if "chest_pain" in symptoms:
        red_flags.append("Chest pain or pressure can be urgent, especially with breathlessness, sweating, nausea, or pain spreading to the arm/jaw/back.")
    if "shortness_breath" in symptoms and ("chest_pain" in symptoms or "cold_sweats" in symptoms):
        red_flags.append("Shortness of breath combined with chest pain or cold sweats may be an emergency.")
    if "reduced_urine" in symptoms and "swelling" in symptoms:
        red_flags.append("Markedly reduced urine output with swelling needs prompt medical assessment.")
    if "radiating_pain" in symptoms and ("chest_pain" in symptoms or "shortness_breath" in symptoms):
        red_flags.append("Pain spreading from the chest with breathing difficulty may be an emergency.")

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
        max_points = sum(item["symptoms"].values()) + sum(rule[2] for rule in item["lab_rules"])
        if points > 0:
            # Match label is a transparent rule coverage bucket, not probability.
            ratio = points / max(max_points, 1)
            match = "Stronger rule match" if ratio >= 0.55 else ("Some rule matches" if ratio >= 0.25 else "Limited rule matches")
            ranked.append({
                "id": item["id"], "name": item["name"], "category": item["category"],
                "specialist": item["specialist"], "match_label": match, "rule_points": points,
                "evidence": evidence[:6], "next_step": item["next_step"]
            })
    ranked.sort(key=lambda x: (x["rule_points"], len(x["evidence"])), reverse=True)
    return {
        "created_at": utc_now(),
        "engine": "DiagNova transparent rule set v0.2",
        "red_flags": red_flags,
        "urgent_notice": bool(red_flags),
        "matches": ranked,
        "input_summary": {
            "age": data.age, "sex": data.sex, "symptoms": [{"key": s, "label": SYMPTOMS[s]} for s in data.symptoms],
            "labs": {k: v for k, v in labs.items() if v is not None}
        },
        "limitations": "Rule points and match labels are not probabilities, diagnoses, or clinically validated risk estimates. Missing values are not treated as zero."
    }


_skin_model = None
_skin_idx_to_class: dict[int, str] | None = None


def load_skin_model():
    global _skin_model, _skin_idx_to_class
    if _skin_model is not None:
        return _skin_model, _skin_idx_to_class
    if not SKIN_MODEL_ENABLED:
        raise HTTPException(status_code=503, detail="Skin model is disabled. Follow README instructions to enable and download it.")
    try:
        import torch
        from torchvision.models import efficientnet_b0
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(repo_id=SKIN_MODEL_REPO, filename="best_model.pth")
        idx_path = hf_hub_download(repo_id=SKIN_MODEL_REPO, filename="class_to_idx.json")
        with open(idx_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        _skin_idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 5)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        _skin_model = model
        return _skin_model, _skin_idx_to_class
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Skin model could not load. Check optional dependencies and model repository. Detail: {type(exc).__name__}")


@app.get("/")
def home():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "api": "DiagNova MVP",
        "skin_model_enabled": SKIN_MODEL_ENABLED,
        "skin_model_loaded": _skin_model is not None,
        "note": "Educational prototype; not for diagnosis, triage, or treatment decisions."
    }


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


@app.post("/api/skin")
async def skin_predict(file: UploadFile = File(...), user=Depends(current_user)):
    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG, or WebP image.")
    raw = await file.read(MAX_IMAGE_BYTES + 1)
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 8 MB or smaller.")
    try:
        from io import BytesIO
        from PIL import Image, UnidentifiedImageError
        image = Image.open(BytesIO(raw)).convert("RGB")
        if image.width < 96 or image.height < 96:
            raise HTTPException(status_code=422, detail="Choose a larger image (at least 96 × 96 pixels).")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=422, detail="The uploaded file could not be read as an image.")

    model, idx_to_class = load_skin_model()
    import torch
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    ranked = sorted(
        [{"label": idx_to_class.get(i, f"class_{i}"), "score": round(float(probs[i]), 4)} for i in range(len(probs))],
        key=lambda x: x["score"], reverse=True
    )
    return {
        "created_at": utc_now(),
        "filename": Path(file.filename or "uploaded-image").name[:120],
        "model": SKIN_MODEL_REPO,
        "input_type": "clinical/macroscopic skin photo",
        "results": ranked,
        "limitations": [
            "Educational research model only; not validated for clinical use, diagnosis, triage, or treatment decisions.",
            "Scores are model outputs, not clinical probabilities or certainty.",
            "Performance varies with image quality, skin tone, and dataset source; this model's model card reports lower performance on a smartphone-photo subset.",
            "The uploaded image is processed in memory and is not saved to the patient record or disk by this endpoint."
        ]
    }


@app.get("/api/evidence")
def evidence(condition: str = "general", user=Depends(current_user)):
    # Live PubMed search. Research is related reading, not evidence that a user has a condition.
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
    # User explicitly saves a result; nothing is persisted automatically.
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


@app.delete("/api/records")
def delete_all_records(confirm: str, user=Depends(current_user)):
    if confirm != "DELETE":
        raise HTTPException(status_code=400, detail="To confirm deletion, send confirm=DELETE.")
    with connect_db() as con:
        cur = con.execute("DELETE FROM records WHERE user_id=?", (user["id"],))
        count = cur.rowcount
    return {"deleted": True, "count": count}


@app.get("/api/symptoms")
def symptoms(user=Depends(current_user)):
    return {"symptoms": [{"key": k, "label": v} for k, v in SYMPTOMS.items()]}


@app.get("/api/knowledge-base")
def knowledge_base(user=Depends(current_user)):
    return {
        "version": "0.2",
        "method": "transparent hand-authored rules; not a trained ML disease predictor",
        "conditions": [{"id": x["id"], "name": x["name"], "category": x["category"], "specialist": x["specialist"]} for x in KNOWLEDGE_BASE]
    }

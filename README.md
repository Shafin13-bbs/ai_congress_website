# DiagNova MVP — FastAPI + web frontend

A small, runnable student MVP that keeps the original DiagNova direction: symptom + lab inputs, independently ranked rule matches, safety red flags, specialist routing, PubMed related reading, skin-image upload, and a private saved-record history.

## What's actually implemented

- FastAPI backend and responsive HTML/CSS/JS frontend served by the same app.
- User registration/sign-in, PBKDF2 password hashing, signed bearer tokens, per-user record ownership.
- SQLite storage for accounts and records.
- Symptom + lab rule engine with visible reasons. This is a **hand-authored deterministic rule set**, not a trained disease model and not a clinical knowledge graph.
- Red-flag messages run before the ranked matches; they are not a complete emergency detector.
- PubMed E-utilities search when online. If PubMed cannot be reached, the app shows that honestly rather than inventing citations.
- User-controlled saving: results are **not saved automatically**. The user must tick the opt-in checkbox. Each record has a delete action with confirmation. “Delete all” requires two explicit confirmations and is never automatic.
- Skin-image endpoint accepts uploaded JPG/PNG/WebP files only; it does not access a camera. Uploaded image bytes are processed in memory and not written to disk by the endpoint. Only a result summary can be saved, if the user opts in.

## Important model limitation

Optional image model: [`RevelaCap/clinical-skin-condition-v1`](https://huggingface.co/RevelaCap/clinical-skin-condition-v1). Its model card documents an EfficientNet-B0 model for clinical/macroscopic skin photographs with five educational classes. It also explicitly says it is **not validated for clinical use** and is **not suitable for patient triage, diagnosis, or treatment decisions**. The card reports macro-F1 0.6527, balanced accuracy 0.6700, and lower macro-F1 on a smartphone-photo subset. Scores shown by this app are model outputs, not calibrated clinical probabilities. This is an educational inference demonstration only.

The skin model is disabled by default to keep installation lighter. Enable it only after installing the optional dependencies and accepting the model limitations. First inference downloads model weights from Hugging Face; internet is required for that first download.

## Run locally (Windows / PowerShell or Git Bash)

1. Install Python 3.11 or 3.12.
2. Open a terminal in this folder.
3. Create and activate a virtual environment:

   **PowerShell**
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **Git Bash**
   ```bash
   python -m venv .venv
   source .venv/Scripts/activate
   ```

4. Install the base dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
5. Create `.env` from `.env.example`, then set a strong random `DIAGNOVA_SECRET_KEY`. If you do not load environment files with a tool, set environment variables in the terminal before launch. (For a simple local demo, the default key works, but do not deploy it.)
6. Start the app:
   ```bash
   uvicorn app:app --reload
   ```
7. Open http://127.0.0.1:8000. The interactive API docs are at http://127.0.0.1:8000/docs.

### Enable the optional skin model

Install PyTorch/torchvision following the official selector for your operating system, then:
```bash
pip install huggingface_hub
```
Set `DIAGNOVA_ENABLE_SKIN_MODEL=1` and optionally `DIAGNOVA_SKIN_MODEL_REPO=RevelaCap/clinical-skin-condition-v1`, then restart the server. The model may require several hundred MB of downloads and may take time on CPU. If loading fails, the API returns a clear error; it never substitutes simulated output.

For PowerShell:
```powershell
$env:DIAGNOVA_ENABLE_SKIN_MODEL="1"
$env:DIAGNOVA_SECRET_KEY="your-long-random-secret"
uvicorn app:app --reload
```

For Git Bash:
```bash
export DIAGNOVA_ENABLE_SKIN_MODEL=1
export DIAGNOVA_SECRET_KEY='your-long-random-secret'
uvicorn app:app --reload
```

## API overview

- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`
- `GET /api/symptoms`
- `POST /api/screen`
- `POST /api/skin` (multipart file upload)
- `GET /api/evidence?condition=...`
- `POST /api/records` (explicit opt-in save)
- `GET /api/records`
- `DELETE /api/records/{record_id}` (owner-checked)
- `DELETE /api/records?confirm=DELETE` (current user's records only)

## Privacy and security notes

This is a **local demo**, not a production medical-record system. Before deployment: use HTTPS, a production-grade session/cookie strategy, CSRF protections if using cookies, rate limiting, email verification, password reset, audit/logging review, database backups, secrets management, data retention policy, security testing, and a clinician/privacy review. Avoid collecting real patient information in a public demo. The browser keeps a bearer token in localStorage for simplicity; replace this with secure HttpOnly cookies before production.

## MVP boundaries / honesty

- No OCR/report extraction or wearable integration is implemented.
- No trained symptom/lab disease prediction model or clinical calibration is implemented.
- Rules are small and hand-authored to demonstrate end-to-end architecture, not to estimate real disease likelihood.
- PubMed results are related research, not proof a person has a condition.
- No real clinical knowledge graph database is used; the displayed flow is an explanatory diagram.
- This software does not diagnose or rule out illness and should not be used for clinical triage or treatment decisions.

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
import shutil
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
LOG_DIR = BASE_DIR / "data" / "logs"
LOG_FILE = LOG_DIR / "submissions.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

if not LOG_FILE.exists():
    pd.DataFrame(columns=[
        "timestamp",
        "prm_id",
        "filename",
        "is_female",
        "has_jio_jacket",
        "has_laminated_jio_promotional_paper",
        "female_confidence",
        "jacket_confidence",
        "paper_confidence",
        "review_required"
    ]).to_csv(LOG_FILE, index=False)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(
    prm_id: str = Form(...),
    photo: UploadFile = File(...)
):
    timestamp = datetime.now().isoformat(timespec="seconds")
    safe_timestamp = timestamp.replace(":", "-")
    save_name = f"{safe_timestamp}_{photo.filename}"
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(photo.file, buffer)

    result = {
        "timestamp": timestamp,
        "prm_id": prm_id,
        "filename": save_name,
        "is_female": "No",
        "has_jio_jacket": "No",
        "has_laminated_jio_promotional_paper": "No",
        "female_confidence": 0.50,
        "jacket_confidence": 0.50,
        "paper_confidence": 0.50,
        "review_required": "Yes"
    }

    df = pd.read_csv(LOG_FILE)
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

    return result
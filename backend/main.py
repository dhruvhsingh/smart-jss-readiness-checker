from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pathlib import Path
from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
import pandas as pd
import io
import os
import requests

from gemini_analyzer import analyze_image_with_gemini

load_dotenv()

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
        "review_required",
        "review_reason",
        "image_width",
        "image_height",
        "image_mode"
    ]).to_csv(LOG_FILE, index=False)


def validate_and_read_image(file_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
        image = Image.open(io.BytesIO(file_bytes))
        return image
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process uploaded image.")

def push_to_google_sheets(payload: dict):
    webhook_url = os.getenv("GOOGLE_SHEETS_WEBHOOK_URL")
    print("push_to_google_sheets() called", flush=True)
    if not webhook_url:
        print("GOOGLE_SHEETS_WEBHOOK_URL not set. Skipping Google Sheets push.")
        return

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=15
        )
        response.raise_for_status()
        print("Google Sheets push successful.", flush=True)
    except Exception as e:
        print(f"Google Sheets push failed: {e}", flush=True)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    prm_id: str = Form(...),
    photo: UploadFile = File(...)
):
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is missing in backend/.env")

    if not prm_id.strip():
        raise HTTPException(status_code=400, detail="PRM ID is required.")

    timestamp = datetime.now().isoformat(timespec="seconds")
    safe_timestamp = timestamp.replace(":", "-")
    save_name = f"{safe_timestamp}_{photo.filename}"
    save_path = UPLOAD_DIR / save_name

    file_bytes = await photo.read()
    image = validate_and_read_image(file_bytes)

    with open(save_path, "wb") as buffer:
        buffer.write(file_bytes)

    width, height = image.size
    mode = image.mode

    gemini_result = analyze_image_with_gemini(str(save_path))

    result = {
        "timestamp": timestamp,
        "prm_id": prm_id,
        "filename": save_name,
        "is_female": gemini_result["is_female"],
        "has_jio_jacket": gemini_result["has_jio_jacket"],
        "has_laminated_jio_promotional_paper": gemini_result["has_laminated_jio_promotional_paper"],
        "female_confidence": gemini_result["female_confidence"],
        "jacket_confidence": gemini_result["jacket_confidence"],
        "paper_confidence": gemini_result["paper_confidence"],
        "review_required": gemini_result["review_required"],
        "review_reason": gemini_result["review_reason"],
        "image_width": width,
        "image_height": height,
        "image_mode": mode
    }

    df = pd.read_csv(LOG_FILE)
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    df.to_csv(LOG_FILE, index=False)

    push_to_google_sheets(result)

    return result
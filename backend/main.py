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

app = FastAPI(
    title="Smart JSS Backend",
    version="1.0.0"
)

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

CSV_COLUMNS = [
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
    "image_mode",
]

if not LOG_FILE.exists():
    pd.DataFrame(columns=CSV_COLUMNS).to_csv(LOG_FILE, index=False)


def ensure_log_file_exists() -> None:
    if not LOG_FILE.exists():
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(LOG_FILE, index=False)


def sanitize_filename(original_name: str, timestamp: str) -> str:
    path = Path(original_name or "upload.jpg")
    stem = path.stem if path.stem else "upload"
    suffix = path.suffix.lower() if path.suffix else ".jpg"

    safe_stem = "".join(
        ch if ch.isalnum() or ch in ("-", "_", " ") else "_"
        for ch in stem
    ).strip()

    safe_stem = safe_stem.replace(" ", "_")
    safe_timestamp = timestamp.replace(":", "-")
    return f"{safe_timestamp}_{safe_stem}{suffix}"


def validate_and_read_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
        image = Image.open(io.BytesIO(file_bytes))
        return image
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read uploaded image.")


def safe_float(value, default=0.0) -> float:
    try:
        return round(float(value), 2)
    except Exception:
        return default


def safe_text(value, default="") -> str:
    if value is None:
        return default
    return str(value).strip()


def safe_error_text(error: Exception, max_len: int = 500) -> str:
    try:
        message = str(error)
    except Exception:
        message = error.__class__.__name__

    if not message:
        message = error.__class__.__name__

    if len(message) > max_len:
        message = message[:max_len] + "..."
    return message


def push_to_google_sheets(payload: dict) -> None:
    webhook_url = os.getenv("GOOGLE_SHEETS_WEBHOOK_URL", "").strip()

    if not webhook_url:
        print("GOOGLE_SHEETS_WEBHOOK_URL not set. Skipping Google Sheets push.")
        return

    try:
        print("push_to_google_sheets() called")
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        print("Google Sheets push successful.")
    except Exception as e:
        print(f"Google Sheets push failed: {safe_error_text(e)}")


@app.get("/")
def root():
    return {"message": "Smart JSS Backend is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    prm_id: str = Form(...),
    photo: UploadFile = File(...),
):
    prm_id = prm_id.strip()

    if not prm_id:
        raise HTTPException(status_code=400, detail="PRM ID is required.")

    if not photo.filename:
        raise HTTPException(status_code=400, detail="Photo file is required.")

    file_bytes = await photo.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded photo is empty.")

    image = validate_and_read_image(file_bytes)
    width, height = image.size
    mode = image.mode

    timestamp = datetime.now().replace(microsecond=0).isoformat()
    save_name = sanitize_filename(photo.filename, timestamp)
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    try:
        gemini_result = analyze_image_with_gemini(file_bytes)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gemini analysis failed: {safe_error_text(e)}"
        )

    result = {
        "timestamp": timestamp,
        "prm_id": prm_id,
        "filename": save_name,
        "is_female": safe_text(gemini_result.get("is_female"), "No"),
        "has_jio_jacket": safe_text(gemini_result.get("has_jio_jacket"), "No"),
        "has_laminated_jio_promotional_paper": safe_text(
            gemini_result.get("has_laminated_jio_promotional_paper"),
            "No"
        ),
        "female_confidence": safe_float(gemini_result.get("female_confidence"), 0.0),
        "jacket_confidence": safe_float(gemini_result.get("jacket_confidence"), 0.0),
        "paper_confidence": safe_float(gemini_result.get("paper_confidence"), 0.0),
        "review_required": safe_text(gemini_result.get("review_required"), "Yes"),
        "review_reason": safe_text(gemini_result.get("review_reason"), ""),
        "image_width": width,
        "image_height": height,
        "image_mode": mode,
    }

    try:
        ensure_log_file_exists()
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
        df.to_csv(LOG_FILE, index=False)
    except Exception as e:
        print(f"CSV log write failed: {safe_error_text(e)}")

    push_to_google_sheets(result)

    return result
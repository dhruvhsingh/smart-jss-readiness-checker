from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from datetime import datetime
from typing import Any, Dict
import csv
import io
import os
import re
import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from gemini_analyzer import analyze_image_with_gemini


app = FastAPI(title="Smart JSS Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
LOG_DIR = DATA_DIR / "logs"
LOG_FILE = LOG_DIR / "submissions.csv"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

CSV_HEADERS = [
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
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writeheader()


def sanitize_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^\w.\- ]+", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name or "upload.jpg"


def normalize_yes_no(value: Any) -> str:
    if isinstance(value, bool):
        return "Yes" if value else "No"

    text = str(value).strip().lower()
    if text in {"yes", "true", "1", "y", "pass", "female", "present"}:
        return "Yes"
    if text in {"no", "false", "0", "n", "fail", "not female", "absent"}:
        return "No"
    return "No"


def normalize_confidence(value: Any) -> float:
    try:
        conf = float(value)
    except Exception:
        conf = 0.0

    if conf < 0:
        conf = 0.0
    if conf > 1:
        if conf <= 100:
            conf = conf / 100.0
        else:
            conf = 1.0

    return round(conf, 2)


def validate_and_read_image(file_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(file_bytes))
        image.verify()
        image = Image.open(io.BytesIO(file_bytes))
        return image
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not process uploaded image.")


def call_gemini_analyzer(file_bytes: bytes, mime_type: str) -> Dict[str, Any]:
    """
    This supports multiple possible function signatures in gemini_analyzer.py
    so your app does not break if that file was written slightly differently.
    """
    last_type_error = None

    attempts = [
        lambda: analyze_image_with_gemini(file_bytes=file_bytes, mime_type=mime_type),
        lambda: analyze_image_with_gemini(image_bytes=file_bytes, mime_type=mime_type),
        lambda: analyze_image_with_gemini(file_bytes, mime_type),
        lambda: analyze_image_with_gemini(file_bytes=file_bytes),
        lambda: analyze_image_with_gemini(file_bytes),
    ]

    for attempt in attempts:
        try:
            result = attempt()
            if isinstance(result, dict):
                return result
        except TypeError as e:
            last_type_error = e
            continue

    raise RuntimeError(
        f"Could not call analyze_image_with_gemini with expected arguments. Last error: {last_type_error}"
    )


def normalize_gemini_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    is_female = normalize_yes_no(raw.get("is_female"))
    has_jio_jacket = normalize_yes_no(raw.get("has_jio_jacket"))
    has_paper = normalize_yes_no(raw.get("has_laminated_jio_promotional_paper"))

    female_confidence = normalize_confidence(raw.get("female_confidence", 0))
    jacket_confidence = normalize_confidence(raw.get("jacket_confidence", 0))
    paper_confidence = normalize_confidence(raw.get("paper_confidence", 0))

    review_required = normalize_yes_no(raw.get("review_required", "No"))
    review_reason = str(raw.get("review_reason", "")).strip()

    if review_required == "Yes" and not review_reason:
        review_reason = "Manual review required."
    if review_required == "No" and not review_reason:
        review_reason = "All checks passed."

    return {
        "is_female": is_female,
        "has_jio_jacket": has_jio_jacket,
        "has_laminated_jio_promotional_paper": has_paper,
        "female_confidence": female_confidence,
        "jacket_confidence": jacket_confidence,
        "paper_confidence": paper_confidence,
        "review_required": review_required,
        "review_reason": review_reason,
    }


def append_to_csv(row: Dict[str, Any]) -> None:
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        writer.writerow({key: row.get(key, "") for key in CSV_HEADERS})


def push_to_google_sheets(payload: Dict[str, Any]) -> None:
    webhook_url = os.getenv("GOOGLE_SHEETS_WEBHOOK_URL", "").strip()
    if not webhook_url:
        print("GOOGLE_SHEETS_WEBHOOK_URL not set. Skipping Google Sheets push.")
        return

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=15,
        )
        response.raise_for_status()
        print("Google Sheets push successful.")
    except Exception as e:
        print(f"Google Sheets push failed: {e}")


@app.get("/")
def root():
    return {"status": "ok", "message": "Smart JSS backend is running."}


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

    file_bytes = await photo.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image = validate_and_read_image(file_bytes)
    width, height = image.size
    mode = image.mode

    original_name = sanitize_filename(photo.filename or "upload.jpg")
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    timestamp_for_filename = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_name = f"{timestamp_for_filename}_{original_name}"
    save_path = UPLOAD_DIR / save_name

    with open(save_path, "wb") as f:
        f.write(file_bytes)

    mime_type = photo.content_type or "image/jpeg"

    try:
        raw_gemini_result = call_gemini_analyzer(file_bytes=file_bytes, mime_type=mime_type)
        gemini_result = normalize_gemini_result(raw_gemini_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini analysis failed: {e}")

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
        "image_mode": mode,
    }

    append_to_csv(result)
    push_to_google_sheets(result)

    return result
import json
import os
import re
from typing import Dict, Any

from google import genai
from google.genai import types


MODEL_NAME = "gemini-2.5-flash"

PROMPT = """
You are a strict Smart JSS readiness checker.

Analyze the uploaded image and evaluate ONLY these three checks:

1. is_female
- Yes only if the clearly visible main person appears female.
- No if male, unclear, no single person, crowd, mannequin, mirror-selfie confusion, or person is not clearly visible.

2. has_jio_jacket
- Yes only if the clearly visible main person is WEARING the blue Jio jacket/vest on their body.
- No if the jacket is absent, kept on a table, hanger, floor, held separately, partially visible without being worn, or not clearly worn.

3. has_laminated_jio_promotional_paper
- Yes only if the clearly visible main person is HOLDING the correct laminated Jio promotional paper in hand.
- No if no paper is present, if the paper is plain/white/internal/training/wrong paper, partially visible, or placed separately instead of being held.

Important evaluation rules:
- The image should ideally contain exactly one main person.
- Ignore the file name completely.
- Judge only from the visual content.
- If the image is blurry, overexposed, too dark, cropped badly, contains a crowd, mirror selfie, or is unclear, set review_required = "Yes".
- If any condition is uncertain, be conservative.

Return ONLY valid JSON in exactly this format:
{
  "is_female": "Yes",
  "has_jio_jacket": "Yes",
  "has_laminated_jio_promotional_paper": "Yes",
  "female_confidence": 0.98,
  "jacket_confidence": 0.97,
  "paper_confidence": 0.96,
  "review_required": "No",
  "review_reason": "Clear single person wearing the blue Jio jacket and holding the correct laminated promotional paper."
}
""".strip()


def _extract_json(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    text = text.replace("```json", "").replace("```", "").strip()

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"Gemini returned non-JSON output: {text[:500]}")

    return json.loads(match.group(0))


def _yes_no(value: Any) -> str:
    return "Yes" if str(value).strip().lower() in {"yes", "true", "1"} else "No"


def _confidence(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        x = 0.50
    return round(max(0.0, min(1.0, x)), 2)


def analyze_image_with_gemini(file_bytes: bytes, mime_type: str = "image/jpeg") -> Dict[str, Any]:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing on the server")

    client = genai.Client(api_key=api_key)

    image_part = types.Part.from_bytes(
        data=file_bytes,
        mime_type=mime_type or "image/jpeg",
    )

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[PROMPT, image_part],
            config=types.GenerateContentConfig(
                temperature=0,
                response_mime_type="application/json",
                thinking_config=types.ThinkingConfig(thinking_level="medium"),
            ),
        )

        data = _extract_json(response.text or "")

        result = {
            "is_female": _yes_no(data.get("is_female")),
            "has_jio_jacket": _yes_no(data.get("has_jio_jacket")),
            "has_laminated_jio_promotional_paper": _yes_no(
                data.get("has_laminated_jio_promotional_paper")
            ),
            "female_confidence": _confidence(data.get("female_confidence")),
            "jacket_confidence": _confidence(data.get("jacket_confidence")),
            "paper_confidence": _confidence(data.get("paper_confidence")),
            "review_required": _yes_no(data.get("review_required")),
            "review_reason": str(data.get("review_reason", "")).strip(),
        }

        # deterministic safeguard
        if min(
            result["female_confidence"],
            result["jacket_confidence"],
            result["paper_confidence"],
        ) < 0.85:
            result["review_required"] = "Yes"
            if result["review_reason"]:
                result["review_reason"] += "; low confidence"
            else:
                result["review_reason"] = "low confidence"

        if not result["review_reason"]:
            result["review_reason"] = (
                "Manual review recommended"
                if result["review_required"] == "Yes"
                else "All checks passed"
            )

        return result

    except Exception as e:
        msg = str(e)
        if len(msg) > 500:
            msg = msg[:500] + "..."
        raise RuntimeError(f"Gemini request failed: {type(e).__name__}: {msg}")
import json
import re

from google import genai


PROMPT = """
You are a strict compliance checker for Jio Smart JSS readiness photos.

Your job is to evaluate ONLY these 3 fields from the uploaded image:

1. is_female
2. has_jio_jacket
3. has_laminated_jio_promotional_paper

This is a strict operational check, not a general image description task.

CRITICAL DECISION RULES:

A. PERSON VISIBILITY RULE
- A compliant photo should show exactly one clearly visible person.
- If no person is visible, multiple people are visible, the image is just an object, or the person is too cropped/blurry/obscured, then:
  - set review_required = "Yes"
  - explain why in review_reason

B. FEMALE RULE
- is_female = "Yes" only if the visible person appears female.
- If there is no clearly visible person, set is_female = "No".
- If gender is ambiguous, set is_female = "No" and set review_required = "Yes".

C. JIO JACKET RULE
- has_jio_jacket = "Yes" only if the blue Jio jacket/vest is clearly being WORN on the torso/body of the visible person.
- If the jacket is only shown alone, hanging, folded, held in hand, placed somewhere, partially visible, or not being worn, then has_jio_jacket = "No".
- Do not mark "Yes" just because a Jio jacket exists somewhere in the image.

D. LAMINATED JIO PROMOTIONAL PAPER RULE
- has_laminated_jio_promotional_paper = "Yes" only if the correct laminated Jio promotional paper is clearly being HELD by the visible person.
- If the paper is absent, wrong, unclear, plain, random, partially visible, or just shown alone without a visible person holding it, then set it to "No".

E. REVIEW RULE
Set review_required = "Yes" if ANY of these are true:
- no clearly visible person
- more than one person
- mirrored selfie / non-standard pose
- very blurry / overexposed / dark image
- heavy crop / partial body / unclear torso area
- gender ambiguous
- jacket visibility unclear
- paper visibility unclear
- any confidence is low or borderline

F. CONFIDENCE RULE
- Return realistic confidence between 0 and 1.
- Do NOT output 1.0 unless the case is extremely clear.
- For object-only images, no-person images, or unclear images, confidence should generally be moderate or low.
- Do not give very high confidence to uncertain conclusions.

Return ONLY valid JSON in exactly this shape:
{
  "is_female": {"value": "Yes", "confidence": 0.95},
  "has_jio_jacket": {"value": "Yes", "confidence": 0.94},
  "has_laminated_jio_promotional_paper": {"value": "Yes", "confidence": 0.97},
  "review_required": "No",
  "review_reason": "clear single person wearing jacket and holding correct laminated paper"
}
""".strip()


def _extract_json(text: str) -> dict:
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in Gemini response.")

    return json.loads(match.group(0))


def _yn(value) -> str:
    return "Yes" if str(value).strip().lower() == "yes" else "No"


def _conf(value) -> float:
    try:
        v = float(value)
    except Exception:
        v = 0.5
    return max(0.0, min(1.0, v))


def analyze_image_with_gemini(image_path: str) -> dict:
    client = genai.Client()

    uploaded = client.files.upload(file=image_path)
    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=[PROMPT, uploaded],
        )

        raw_text = response.text or ""
        parsed = _extract_json(raw_text)

        is_female = _yn(parsed["is_female"]["value"])
        has_jio_jacket = _yn(parsed["has_jio_jacket"]["value"])
        has_paper = _yn(parsed["has_laminated_jio_promotional_paper"]["value"])

        female_conf = _conf(parsed["is_female"]["confidence"])
        jacket_conf = _conf(parsed["has_jio_jacket"]["confidence"])
        paper_conf = _conf(parsed["has_laminated_jio_promotional_paper"]["confidence"])

        review_required = _yn(parsed.get("review_required", "Yes"))
        review_reason = str(parsed.get("review_reason", "")).strip()

        # Deterministic safeguard:
        # if any confidence is below 0.85, force manual review
        if min(female_conf, jacket_conf, paper_conf) < 0.85:
            review_required = "Yes"
            if review_reason:
                review_reason += "; low confidence"
            else:
                review_reason = "low confidence"

        return {
            "is_female": is_female,
            "female_confidence": female_conf,
            "has_jio_jacket": has_jio_jacket,
            "jacket_confidence": jacket_conf,
            "has_laminated_jio_promotional_paper": has_paper,
            "paper_confidence": paper_conf,
            "review_required": review_required,
            "review_reason": review_reason or "manual review recommended",
        }
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass
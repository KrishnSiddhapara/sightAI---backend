import json
import logging
import time
import socket
from typing import List, Tuple, Optional
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
# pyrefly: ignore [missing-import]
from google.genai.errors import APIError

from services.schemas import GroundedAnalysisResult
from utils.config import MAX_OUTPUT_TOKENS
from utils.json_utils import clean_json_text
from utils.image_validation import encode_vlm_image_part

logger = logging.getLogger(__name__)

GROUNDED_VISION_SYSTEM_PROMPT = """You are a world-class Computer Vision & Physical Object Verification AI.
Your absolute top priority is VISUAL ACCURACY, GROUNDING, and ZERO HALLUCINATION.

Follow this MANDATORY step-by-step visual verification pipeline before producing structured output:

STEP 1: FULL IMAGE SCAN & ENTITY DETECTION
- Scan the entire image for distinct, visually observable physical entities.
- Distinguish between real physical objects in the 3D scene vs reflections in mirrors/glass, shadows on surfaces, or pictures shown inside posters/TV screens/paintings.
- DO NOT count reflections, shadows, or images shown inside screens/posters/photos as real physical objects.

STEP 2: PHYSICAL INSTANCE COUNTING & DEDUPLICATION
- Count distinct physical instances.
- DO NOT double-count an object under different names (e.g. if an item is a 'car', do NOT list it as 'car' and also as 'vehicle' or 'automobile').
- Standardize object category names to simple lowercase terms (e.g. 'person', 'car', 'dog', 'chair', 'bottle', 'bicycle').
- For overlapping or partially hidden objects:
  * Count ONLY clearly distinguishable physical instances under 'confirmed_count'.
  * If an object is partially visible but cannot be confirmed, increment 'uncertain_count' and explain the reason in 'uncertainty_reason'. Do NOT guess or pad the confirmed count.

STEP 3: INDEPENDENT PERSON-BY-PERSON & INSTANCE ATTRIBUTE EXTRACTION
- EVERY person instance (person_1, person_2, person_3, etc.) MUST be analyzed independently based strictly on their visible spatial region.
- NEVER create a generic description and copy it to multiple people.
- If Person 1 wears a red shirt and Person 2 wears a blue jacket, Person 1 MUST have clothing_color='red' and Person 2 MUST have clothing_color='blue'. NEVER share or copy attributes across entities!
- If an attribute (e.g. exact clothing, pose, action, age, brand) is not clearly visible, return "not clearly visible" or "unknown". DO NOT guess!

STEP 4: GROUNDED BOUNDING BOX LOCALIZATION
- For EVERY confidently identified physical instance, provide bounding box coordinates in 'bounding_box': {"x_min": int, "y_min": int, "x_max": int, "y_max": int}.
- Use normalized 0 to 1000 integer coordinates where:
  * x_min: Leftmost edge of the object (0 = left boundary of image, 1000 = right boundary of image)
  * y_min: Topmost edge of the object (0 = top boundary of image, 1000 = bottom boundary of image)
  * x_max: Rightmost edge of the object (0 to 1000)
  * y_max: Bottommost edge of the object (0 to 1000)
- Ensure x_min < x_max and y_min < y_max.
- The bounding box MUST tightly surround that specific object instance.
- If localization for an instance is uncertain, unconfirmed, or severely occluded, set 'bounding_box': null. NEVER invent fake or estimated coordinates!

STEP 5: CONCISE SCENE CLASSIFICATION
- Classify the primary scene category as a CONCISE 1-3 word title (e.g. 'Office', 'Classroom', 'Street', 'Living Room', 'Kitchen', 'Outdoor Park', 'Restaurant', 'Sports Field', 'Beach', 'Warehouse', 'Document').
- NEVER return a full sentence or description under 'category'. Put detailed descriptions in 'environment', 'primary_activity', and 'summary'.

STEP 6: IMAGE-WIDE CONSISTENCY CROSS-CHECK
Before returning the final structured JSON, cross-check:
A. Is every reported object actually visible in the image?
B. Is the confirmed count equal to the number of distinct physical instances?
C. Are attributes assigned strictly to their correct owner entity?
D. Were any shadows, reflections, or photos inside screens mistakenly counted?
E. Does every bounding box surround the exact target instance and remain inside valid [0, 1000] boundaries?
F. Is scene category a concise 1-3 word title?
"""

def sanitize_scene_category(result: GroundedAnalysisResult) -> GroundedAnalysisResult:
    """
    Ensures scene.category is a clean, concise 1-3 word title.
    Normalizes long sentence outputs or punctuation artifacts.
    """
    if result.scene and result.scene.category:
        raw_cat = result.scene.category.strip().strip('*"`\'')
        if '.' in raw_cat or ',' in raw_cat or len(raw_cat.split()) > 3:
            cleaned_words = [w.strip('.,;:') for w in raw_cat.split() if w.strip('.,;:')]
            raw_cat = " ".join(cleaned_words[:3])
        result.scene.category = raw_cat.title() if raw_cat else "General Scene"
    return result

def sanitize_bounding_boxes(result: GroundedAnalysisResult) -> GroundedAnalysisResult:
    """
    Validates and sanitizes bounding boxes across all object instances.
    Auto-repairs inverted coordinates (x_min > x_max or y_min > y_max) using min/max,
    ensures boundaries are within [0, 1000], and discards boxes with zero/tiny area (< 5px).
    """
    for category in result.objects:
        for instance in category.instances:
            bbox = instance.bounding_box
            if bbox is not None:
                try:
                    x1 = max(0, min(1000, int(bbox.x_min)))
                    y1 = max(0, min(1000, int(bbox.y_min)))
                    x2 = max(0, min(1000, int(bbox.x_max)))
                    y2 = max(0, min(1000, int(bbox.y_max)))

                    x_min = min(x1, x2)
                    x_max = max(x1, x2)
                    y_min = min(y1, y2)
                    y_max = max(y1, y2)

                    # Ensure box has minimum visible dimension (at least 5 units in 1000 scale)
                    if (x_max - x_min) >= 5 and (y_max - y_min) >= 5:
                        bbox.x_min = x_min
                        bbox.y_min = y_min
                        bbox.x_max = x_max
                        bbox.y_max = y_max
                    else:
                        logger.warning(f"Discarding zero-area bounding box for instance {instance.id}: [{x_min}, {y_min}, {x_max}, {y_max}]")
                        instance.bounding_box = None
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid bounding box types for instance {instance.id}: {e}")
                    instance.bounding_box = None
    return result

from typing import List, Tuple, Optional, Union
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import APIError

from services.schemas import GroundedAnalysisResult
from utils.config import MAX_OUTPUT_TOKENS, PRIMARY_VLM_MODEL, FALLBACK_VLM_MODEL, ANALYSIS_TIMEOUT
from utils.json_utils import clean_json_text
from utils.image_validation import encode_vlm_image_part

logger = logging.getLogger(__name__)

def analyze_image_grounded(image: Union[Image.Image, types.Part], api_key: str) -> GroundedAnalysisResult:
    """
    Sends image or pre-encoded Part to Gemini VLM with structured Pydantic schema (GroundedAnalysisResult).
    Uses PRIMARY_VLM_MODEL and FALLBACK_VLM_MODEL with explicit timeouts and controlled single retries.
    """
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("API_KEY_ERROR: Gemini API key (VLM_API_KEY) is missing or unconfigured.")

    if isinstance(image, types.Part):
        image_part = image
    elif isinstance(image, Image.Image):
        img_bytes, mime_type, _, _ = encode_vlm_image_part(image)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
    else:
        raise ValueError("INVALID_IMAGE: Provided input is not a valid image or Part object.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(ANALYSIS_TIMEOUT * 1000))
    )

    config = types.GenerateContentConfig(
        system_instruction=GROUNDED_VISION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=GroundedAnalysisResult,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    models_to_try = ["gemini-2.5-flash"]

    last_exception = None
    t0 = time.perf_counter()

    for model_name in models_to_try:
        for attempt in range(1, 2 + 1):  # Max 1 retry per model
            try:
                logger.info(f"[VLM_CALL] Invoking model='{model_name}' (attempt {attempt}/2)")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image_part, "Perform strict step-by-step visual verification, physical instance counting, independent attribute analysis, bounding box localization, and scene classification on this image."],
                    config=config,
                )

                t_elapsed = time.perf_counter() - t0
                if not response or not response.text:
                    raise ValueError(f"GEMINI_BAD_RESPONSE: Received empty response text from Gemini Vision API model '{model_name}'.")

                raw_text = clean_json_text(response.text)
                try:
                    data = json.loads(raw_text)
                    raw_result = GroundedAnalysisResult.model_validate(data)
                except Exception as json_err:
                    logger.warning(f"JSON schema parsing failed on model '{model_name}': {json_err}")
                    raise ValueError(f"SCHEMA_VALIDATION_ERROR: Model '{model_name}' response did not match expected GroundedAnalysisResult schema.")

                sanitized = sanitize_bounding_boxes(raw_result)
                final_res = sanitize_scene_category(sanitized)
                logger.info(f"[VLM_SUCCESS] Grounded VLM analysis completed with '{model_name}' in {t_elapsed:.3f}s (objects={len(final_res.objects)}, scene={final_res.scene.category}).")
                return final_res

            except (socket.gaierror, ConnectionError) as net_err:
                last_exception = net_err
                logger.warning(f"[VLM_WARN] Network connection error on model '{model_name}' (attempt {attempt}/2): {net_err}")
                if attempt < 2:
                    time.sleep(1.5)
                    continue
                break

            except TimeoutError as to_err:
                last_exception = to_err
                logger.warning(f"[VLM_WARN] Timeout error on model '{model_name}' (attempt {attempt}/2): {to_err}")
                if attempt < 2:
                    time.sleep(1.5)
                    continue
                break

            except APIError as api_err:
                last_exception = api_err
                err_msg = str(api_err).lower()
                code = getattr(api_err, "code", None)

                if code == 401 or "api key" in err_msg or "unauthorized" in err_msg:
                    raise ValueError(f"API_KEY_ERROR: Invalid Gemini API key provided. ({api_err.message})")

                if code == 400 and "part exceeded" not in err_msg:
                    raise ValueError(f"INVALID_REQUEST: {api_err.message}")

                is_capacity = code in [503, 429] or "capacity" in err_msg or "overloaded" in err_msg or "rate" in err_msg
                logger.warning(f"[VLM_WARN] Gemini API error on model '{model_name}' (code={code}, attempt {attempt}/2): {api_err}")
                if is_capacity and attempt < 2:
                    time.sleep(1.5)
                    continue
                break

            except Exception as e:
                last_exception = e
                logger.warning(f"[VLM_WARN] Exception on model '{model_name}' (attempt {attempt}/2): {e}")
                if attempt < 2:
                    time.sleep(1.0)
                    continue
                break

    if last_exception:
        err_str = str(last_exception)
        err_lower = err_str.lower()
        if "503" in err_lower or "capacity" in err_lower or "unavailable" in err_lower or "overloaded" in err_lower:
            raise ConnectionError("GEMINI_CAPACITY: Google Gemini Vision model is currently at maximum server capacity (503). Please wait a few seconds and try again.")
        elif "timeout" in err_lower or isinstance(last_exception, TimeoutError):
            raise TimeoutError(f"GEMINI_TIMEOUT: Gemini VLM request timed out after {ANALYSIS_TIMEOUT}s. Please try again.")
        elif "getaddrinfo" in err_lower or isinstance(last_exception, (socket.gaierror, ConnectionError)):
            raise ConnectionError("NETWORK_ERROR: Network connection failed during Gemini API call. Please check your internet connection.")
        elif "api_key_error" in err_lower or "invalid_request" in err_lower or "schema_validation_error" in err_lower or "gemini_bad_response" in err_lower:
            raise ValueError(err_str)
        else:
            raise ValueError(f"INTERNAL_ERROR: Failed to perform visual analysis ({err_str}).")

    raise ValueError("INTERNAL_ERROR: Unexpected error during visual analysis.")

def parse_vlm_response(response_text: str) -> Tuple[List[str], str]:
    """
    Legacy parser function maintained for backwards compatibility.
    """
    objects_list: List[str] = []
    description: str = ""
    try:
        lines = [line.strip() for line in response_text.strip().split("\n")]
        current_section = None
        for line in lines:
            if not line:
                continue
            lower_line = line.lower()
            if "objects:" in lower_line or "objects detected:" in lower_line:
                current_section = "objects"
                continue
            elif "description:" in lower_line:
                current_section = "description"
                continue
            if current_section == "objects":
                clean_line = line
                if line.startswith("-") or line.startswith("*"):
                    clean_line = line[1:].strip()
                elif "." in line:
                    parts = line.split(".", 1)
                    if parts[0].strip().isdigit():
                        clean_line = parts[1].strip()
                objects_list.append(clean_line)
            elif current_section == "description":
                description += line + " "
        description = description.strip()
    except Exception as e:
        logger.warning(f"Error while parsing legacy VLM response: {e}")
    return objects_list, description

def analyze_image(image: Image.Image, api_key: str) -> str:
    """
    Legacy plain-text analysis function maintained for backwards compatibility.
    """
    if not api_key:
        raise ValueError("API key is missing.")

    client = genai.Client(api_key=api_key)
    prompt = """Analyze this image and identify the important visible objects. Return Objects list and Description."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt]
    )

    if not response or not response.text:
        raise ValueError("Received an empty response from the VLM API.")

    return response.text

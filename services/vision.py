import json
import logging
import time
import socket
from typing import List, Tuple, Optional, Union
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
# pyrefly: ignore [missing-import]
from google.genai.errors import APIError

from services.schemas import GroundedAnalysisResult
from utils.config import MAX_OUTPUT_TOKENS, PRIMARY_VLM_MODEL, FALLBACK_VLM_MODEL, ANALYSIS_TIMEOUT
from utils.json_utils import clean_json_text, normalize_grounded_analysis, format_pydantic_validation_error
from utils.image_validation import encode_vlm_image_part

logger = logging.getLogger(__name__)

GROUNDED_VISION_SYSTEM_PROMPT = """You are a world-class Computer Vision & Physical Object Verification AI.
Your absolute top priority is VISUAL ACCURACY, GROUNDING, and ZERO HALLUCINATION.

Follow this MANDATORY step-by-step visual verification pipeline before producing structured output:

STEP 1: FULL IMAGE SCAN & ENTITY DETECTION
- Scan the entire image for distinct, visually observable physical entities.
- Distinguish between real physical objects in the 3D scene vs reflections in mirrors/glass, shadows on surfaces, or pictures shown inside posters/TV screens/paintings.
- DO NOT count reflections, shadows, or images shown inside screens/posters/photos as real physical objects.
- OBJECT CATEGORY CONTROL: Detect up to a maximum of 15 primary, clearly visible physical object categories. Do NOT produce an endless list of trivial background micro-objects (such as individual leaves, tiny pebbles, or distant background specks).

STEP 2: PHYSICAL INSTANCE COUNTING & DEDUPLICATION
- Count distinct physical instances.
- DO NOT double-count an object under different names (e.g. if an item is a 'car', do NOT list it as 'car' and also as 'vehicle' or 'automobile').
- Standardize object category names to simple lowercase terms (e.g. 'person', 'car', 'dog', 'chair', 'bottle', 'bicycle', 'phone', 'pen', 'cup').
- For overlapping or partially hidden objects:
  * Count ONLY clearly distinguishable physical instances under 'confirmed_count'.
  * If an object is partially visible but cannot be confirmed, increment 'uncertain_count' and explain the reason in 'uncertainty_reason'. Do NOT guess or pad the confirmed count.

STEP 3: INDEPENDENT PERSON-BY-PERSON & INSTANCE ATTRIBUTE EXTRACTION
- EVERY person instance (person_1, person_2, person_3, etc.) MUST be analyzed independently based strictly on their visible spatial region.
- NEVER create a generic description and copy it to multiple people.
- ATTRIBUTES MUST BE CONCISE 1-4 WORD PHRASES (e.g. clothing='red t-shirt and jeans', pose='standing', action='waving'). NEVER write long prose, paragraphs, explanations, or commentary inside attribute fields!
- If an attribute (e.g. exact clothing, pose, action, age, brand) is not clearly visible, return "not clearly visible" or "unknown". DO NOT guess!

STEP 4: GROUNDED BOUNDING BOX LOCALIZATION & ACCURACY
- For EVERY confidently identified physical object instance, provide bounding box coordinates in 'bounding_box': {"x_min": float, "y_min": float, "x_max": float, "y_max": float} in normalized 0 to 1000 scale:
  * x_min = Leftmost horizontal edge of the object (0 = left image boundary, 1000 = right image boundary)
  * y_min = Topmost vertical edge of the object (0 = top image boundary, 1000 = bottom image boundary)
  * x_max = Rightmost horizontal edge of the object (0 to 1000, MUST be > x_min)
  * y_max = Bottommost vertical edge of the object (0 to 1000, MUST be > y_min)
- CRITICAL BOUNDING BOX ACCURACY RULES:
  1. TIGHT FIT: The bounding box MUST tightly surround the visible physical extent of that specific object. Do NOT include unnecessary surrounding background space, and avoid oversized boxes.
  2. SPATIAL AXIS ALIGNMENT: x_min and x_max measure horizontal left-to-right position; y_min and y_max measure vertical top-to-bottom position.
  3. INDEPENDENT INSTANCES: Each physical instance (e.g., person_1, person_2, bottle_1, bottle_2) MUST have its own separate, independently calculated bounding box. NEVER copy, duplicate, or mirror bounding box coordinates across different objects.
  4. SMALL OBJECTS: For small items (e.g. pens, phones, cups, bottles, remotes, computer mice, small electronics), detect them carefully and ensure the box is compact and tightly fitted to the visible object bounds. Do not omit small objects simply because they are small.
  5. OVERLAPPING OBJECTS: For overlapping objects (e.g., multiple people sitting together, items on a table), preserve separate, distinct bounding boxes for each physical instance.
  6. OCCLUSION & BOUNDARIES: Enclose only the visible physical extent of the object. Do not invent bounding boxes for non-existent objects or areas outside the image frame.
  7. UNCERTAIN LOCALIZATION: If localization for an instance is uncertain, unconfirmed, or severely occluded, set 'bounding_box': null. NEVER return fake or estimated coordinates!

STEP 5: SCENE UNDERSTANDING & DESCRIPTION
- Provide detailed scene details under 'environment', 'primary_activity', and 'summary'.
- Keep descriptions concise (1-2 clear factual sentences, max 25 words).
- Ground all scene observations strictly in visible features.

STEP 6: STRICT CANONICAL JSON OUTPUT FORMAT
Return ONLY ONE valid raw JSON object matching this exact schema:
{
  "objects": [
    {
      "name": "person",
      "confirmed_count": 1,
      "uncertain_count": 0,
      "instances": [
        {
          "id": "person_1",
          "attributes": {
            "clothing": "red t-shirt and jeans",
            "clothing_color": "red",
            "pose": "standing",
            "action": "waving",
            "accessories": "none visible",
            "object_color": "not clearly visible",
            "type_or_subtype": "unknown",
            "visible_details": "none noted"
          },
          "bounding_box": {
            "x_min": 100.0,
            "y_min": 150.0,
            "x_max": 300.0,
            "y_max": 800.0
          },
          "uncertainty_reason": null
        }
      ]
    }
  ],
  "scene": {
    "environment": "Paved street with sidewalk",
    "primary_activity": "People walking outdoors",
    "summary": "Outdoor daytime street scene."
  },
  "overall_summary": "Executive visual analysis summary."
}
CRITICAL RULES:
- Do NOT use Markdown code block fences (do NOT use ```json or ```).
- Do NOT output any preamble, text, explanation, or intro before the JSON object.
- Do NOT output any postamble, notes, or text after the JSON object.
- Use exact field names defined by the schema. Do NOT rename fields.
- Return a complete and valid JSON object.
"""

from utils.coordinate_utils import sanitize_box

def sanitize_bounding_boxes(result: GroundedAnalysisResult) -> GroundedAnalysisResult:
    """
    Validates and sanitizes bounding boxes across all object instances.
    Auto-repairs inverted coordinates (x_min > x_max or y_min > y_max) using min/max,
    ensures boundaries are within [0, 1000], and discards boxes with zero/tiny area (< 2 units in 1000 scale).
    """
    for category in result.objects:
        for instance in category.instances:
            bbox = instance.bounding_box
            if bbox is not None:
                sanitized = sanitize_box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, min_size_px_in_1000=5.0)
                if sanitized:
                    bbox.x_min = sanitized['x_min']
                    bbox.y_min = sanitized['y_min']
                    bbox.x_max = sanitized['x_max']
                    bbox.y_max = sanitized['y_max']
                else:
                    logger.warning(f"Discarding tiny/zero-area bounding box for instance {instance.id}")
                    instance.bounding_box = None
    return result

def analyze_image_grounded(image: Union[Image.Image, types.Part], api_key: str) -> GroundedAnalysisResult:
    """
    Sends image or pre-encoded Part to Gemini VLM with structured Pydantic schema (GroundedAnalysisResult).
    Uses PRIMARY_VLM_MODEL and FALLBACK_VLM_MODEL with explicit timeouts, normalization, and field-level validation logging.
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
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    models_to_try = [PRIMARY_VLM_MODEL]
    if FALLBACK_VLM_MODEL and FALLBACK_VLM_MODEL not in models_to_try:
        models_to_try.append(FALLBACK_VLM_MODEL)

    last_exception = None
    t0 = time.perf_counter()

    for model_name in models_to_try:
        for attempt in range(1, 2 + 1):  # Max 2 attempts per model
            try:
                # On retry after MAX_TOKENS, use a hyper-concise user prompt instruction
                if attempt == 1:
                    user_prompt = "Perform strict visual verification, physical instance counting, independent attribute analysis, bounding box localization, and scene classification on this image."
                else:
                    user_prompt = "CONCISE RETRY: Previous response hit token limit. Return the exact GroundedAnalysisResult JSON schema using EXTREMELY concise 1-2 word attribute values, limit object categories to top 10 most prominent items, and keep scene summary under 15 words. Do NOT list trivial background micro-objects or write prose descriptions inside attributes."

                logger.info(
                    f"[VLM_CONFIG] model='{model_name}' | attempt={attempt}/2 | max_output_tokens={MAX_OUTPUT_TOKENS} | "
                    f"response_mime='application/json' | tools_enabled=False | automatic_function_calling=False"
                )
                logger.info(f"[VLM_CALL] Invoking model='{model_name}' (attempt {attempt}/2)")
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image_part, user_prompt],
                    config=config,
                )

                t_elapsed = time.perf_counter() - t0

                # Extract metadata from candidate response
                candidates = getattr(response, "candidates", None) or []
                cand_count = len(candidates)
                finish_reason_name = "UNKNOWN"
                if candidates:
                    cand_0 = candidates[0]
                    fr = getattr(cand_0, "finish_reason", None)
                    if hasattr(fr, "name"):
                        finish_reason_name = fr.name
                    elif fr is not None:
                        finish_reason_name = str(fr)

                raw_resp_text = response.text if response and hasattr(response, "text") else None
                resp_len = len(raw_resp_text) if raw_resp_text else 0
                resp_preview = (raw_resp_text[:400] + "...") if raw_resp_text and len(raw_resp_text) > 400 else (raw_resp_text or "<empty>")

                logger.info(
                    f"[VLM_RESPONSE_METADATA] model='{model_name}' | candidate_count={cand_count} | "
                    f"finish_reason='{finish_reason_name}' | response_mime='application/json' | "
                    f"max_output_tokens={MAX_OUTPUT_TOKENS} | response_length={resp_len} | preview='{resp_preview[:200]}'"
                )

                # Check for empty response or safety blocks
                if not raw_resp_text:
                    if "SAFETY" in finish_reason_name.upper():
                        raise ValueError(f"SAFETY_BLOCKED: Model '{model_name}' response was blocked by safety filters (finish_reason={finish_reason_name}).")
                    raise ValueError(f"EMPTY_MODEL_RESPONSE: Received empty candidate response text from Gemini Vision API model '{model_name}' (finish_reason={finish_reason_name}).")

                # Check for response truncation
                if finish_reason_name.upper() == "MAX_TOKENS":
                    logger.warning(f"[VLM_TRUNCATED] Model output truncated on attempt {attempt}/2: model='{model_name}', finish_reason=MAX_TOKENS, len={resp_len}")
                    if attempt == 1:
                        # Continue loop to execute concise retry on attempt 2
                        continue
                    raise ValueError(f"MODEL_OUTPUT_TRUNCATED: Model '{model_name}' response was truncated before completion (finish_reason=MAX_TOKENS).")

                # Step A: Clean & Extract JSON
                extracted_json_text = clean_json_text(raw_resp_text)
                
                # Parse JSON safely with controlled single-attempt recovery
                data = None
                try:
                    data = json.loads(extracted_json_text)
                except Exception as json_parse_err:
                    # Single controlled recovery attempt: search raw text directly for first '{' and last '}'
                    first_b = raw_resp_text.find('{')
                    last_b = raw_resp_text.rfind('}')
                    if first_b != -1 and last_b != -1 and first_b < last_b:
                        fallback_slice = raw_resp_text[first_b:last_b+1].strip()
                        try:
                            data = json.loads(fallback_slice)
                            logger.info(f"[JSON_RECOVERY_SUCCESS] Successfully recovered JSON object from raw response on model '{model_name}'")
                        except Exception:
                            data = None

                    if data is None:
                        logger.error(
                            f"[INVALID_JSON] Failed to parse JSON output on model '{model_name}'. "
                            f"\nError: {json_parse_err}"
                            f"\nFinish reason: {finish_reason_name}"
                            f"\nResponse length: {resp_len}"
                            f"\nRaw response preview:\n{resp_preview}"
                        )
                        raise ValueError(f"INVALID_JSON: Model '{model_name}' returned malformed JSON output.")

                # Step B: Controlled Normalization Layer
                normalized_data = normalize_grounded_analysis(data)

                # Step C: GroundedAnalysisResult Pydantic Validation
                try:
                    raw_result = GroundedAnalysisResult.model_validate(normalized_data)
                except Exception as json_err:
                    detailed_err = format_pydantic_validation_error(json_err)
                    logger.error(
                        f"[SCHEMA_VALIDATION_ERROR] Model '{model_name}' response failed GroundedAnalysisResult validation."
                        f"\nField-level details:\n{detailed_err}"
                        f"\nNormalized JSON snippet:\n{json.dumps(normalized_data)[:1000]}"
                    )
                    raise ValueError(f"SCHEMA_VALIDATION_ERROR: Model '{model_name}' response did not match expected GroundedAnalysisResult schema.")

                final_res = sanitize_bounding_boxes(raw_result)
                logger.info(f"[VLM_SUCCESS] Grounded VLM analysis completed with '{model_name}' in {t_elapsed:.3f}s (objects={len(final_res.objects)}).")
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

            except ValueError as val_err:
                # ValueErrors like SCHEMA_VALIDATION_ERROR or INVALID_JSON from inside the try block
                last_exception = val_err
                logger.warning(f"[VLM_WARN] Validation/schema error on model '{model_name}' (attempt {attempt}/2): {val_err}")
                if attempt < 2:
                    time.sleep(1.0)
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

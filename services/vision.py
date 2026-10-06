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
- OBJECT & FIGURINE RECOGNITION: Accurately identify specific decorative items, figurines, statuettes, and sculptures (e.g. 'elephant figurine', 'animal statue', 'plant', 'computer mouse', 'smartwatch', 'towel') by their actual visually observable shape and characteristics. Do NOT use lazy generic names (such as 'round object', 'thing', or 'item') when the object has a distinct form like an elephant figurine or carved statue.

STEP 2: PHYSICAL INSTANCE COUNTING & DEDUPLICATION
- Count distinct physical instances.
- DO NOT double-count an object under different names (e.g. if an item is an 'elephant figurine', do NOT list it as 'elephant figurine' AND also as 'round object' or 'candle').
- Standardize object category names to simple lowercase terms (e.g. 'person', 'car', 'dog', 'chair', 'bottle', 'bicycle', 'phone', 'elephant figurine', 'computer mouse', 'plant', 'towel').
- For overlapping or partially hidden objects:
  * Count ONLY clearly distinguishable physical instances under 'confirmed_count'.
  * If an object is partially visible but cannot be confirmed, increment 'uncertain_count' and explain the reason in 'uncertainty_reason'. Do NOT guess or pad the confirmed count.

STEP 3: INDEPENDENT PERSON-BY-PERSON & INSTANCE ATTRIBUTE EXTRACTION
- EVERY person or object instance (person_1, elephant figurine_1, mouse_1, etc.) MUST be analyzed independently based strictly on their visible spatial region.
- NEVER create a generic description and copy it to multiple instances.
- ATTRIBUTES MUST BE CONCISE 1-4 WORD PHRASES (e.g. clothing='red t-shirt and jeans', pose='standing', action='waving', object_color='white marble'). NEVER write long prose, paragraphs, explanations, or commentary inside attribute fields!
- If an attribute (e.g. exact clothing, pose, action, age, brand) is not clearly visible, return "not clearly visible" or "unknown". DO NOT guess!

STEP 4: GROUNDED BOUNDING BOX LOCALIZATION & ACCURACY
- For EVERY confidently identified physical object instance, provide bounding box coordinates in 'bounding_box': {"x_min": float, "y_min": float, "x_max": float, "y_max": float} in normalized 0 to 1000 scale:
  * x_min = Leftmost horizontal edge of the object (0 = left image boundary, 1000 = right image boundary)
  * y_min = Topmost vertical edge of the object (0 = top image boundary, 1000 = bottom image boundary)
  * x_max = Rightmost horizontal edge of the object (0 to 1000, MUST be > x_min)
  * y_max = Bottommost vertical edge of the object (0 to 1000, MUST be > y_min)
- COORDINATE SYSTEM CALIBRATION: The 0-1000 coordinate space maps linearly across the FULL image rectangle. The user will tell you the exact pixel dimensions (W×H) in their request. Use that to calibrate your spatial awareness:
  * An object at the exact center of the image = x_min~400, y_min~400, x_max~600, y_max~600
  * An object in the top-left corner = x_min~0, y_min~0
  * An object in the bottom-right corner = x_max~1000, y_max~1000
- CRITICAL BOUNDING BOX ACCURACY RULES:
  1. TIGHT FIT: The bounding box MUST tightly surround ONLY the visible physical extent of that specific object instance. Do NOT include surrounding background, empty desk/table space, cords, wires, or shadows. Oversized boxes are WRONG.
  2. SPATIAL AXIS ALIGNMENT: x_min and x_max measure HORIZONTAL left-to-right position; y_min and y_max measure VERTICAL top-to-bottom position. NEVER swap or transpose X↔Y axes! If an object is horizontally left, its x_min should be low (~0-200). If it is vertically high, its y_min should be low (~0-200).
  3. INDEPENDENT INSTANCES: Each distinct physical instance MUST have its own separately calculated bounding box enclosing that exact item. NEVER copy, duplicate, or share coordinates between different objects.
  4. SMALL & TABLETOP OBJECTS: For small items (figurines, pens, phones, cups, mice, watches), detect each individual piece and ensure the box is compact and tightly fitted. Do NOT place boxes on empty spaces, shadows, or cables.
  5. ONE BOX PER PHYSICAL ITEM: Never generate multiple overlapping boxes for the same physical item under different category names. Each real-world object gets exactly ONE bounding box.
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

import re
from pydantic import BaseModel, Field

from services.schemas import BoundingBox
from utils.coordinate_utils import (
    sanitize_box,
    calculate_iou,
    calculate_containment,
    calculate_center_distance,
    calculate_box_center,
    validate_bbox_coords,
    parse_gemini_box_2d,
    box_2d_to_dict
)

GENERIC_OBJECT_NAMES = {"round object", "object", "thing", "item", "shape", "blob", "miscellaneous", "unknown"}

# Safe, unambiguous category aliases to prevent aggressive false matches
SAFE_CATEGORY_ALIASES = {
    "cellphone": "phone",
    "mobile phone": "phone",
    "smartphone": "phone",
    "telephone": "phone",
    "automobile": "car",
    "vehicle": "car",
    "bicycle": "bike",
    "couch": "sofa",
    "eyeglasses": "glasses",
    "sunglasses": "glasses",
    "spectacles": "glasses",
    "computer mouse": "mouse",
    "wristwatch": "watch",
    "smartwatch": "watch",
    "elephant figurine": "elephant",
    "elephant statue": "elephant",
}

def normalize_label(label: str) -> str:
    """Normalizes object label for safe exact or alias comparison."""
    if not label:
        return ""
    cleaned = re.sub(r'[^a-z0-9\s]', '', label.lower().strip())
    tokens = cleaned.split()
    normalized_tokens = []
    for t in tokens:
        # Safe singularization for simple plural endings
        if len(t) > 3 and t.endswith('s') and not t.endswith(('ss', 'us', 'is')):
            normalized_tokens.append(t[:-1])
        else:
            normalized_tokens.append(t)
    base = " ".join(normalized_tokens)
    return SAFE_CATEGORY_ALIASES.get(base, base)

def labels_match(label1: str, label2: str) -> bool:
    """
    Strict category match: exact normalized match or safe explicit alias match only.
    Prevents unrelated labels (e.g. 'cup' vs 'bottle', 'phone' vs 'laptop') from matching.
    """
    n1 = normalize_label(label1)
    n2 = normalize_label(label2)
    if not n1 or not n2:
        return False
    return n1 == n2

def sanitize_bounding_boxes(result: GroundedAnalysisResult) -> GroundedAnalysisResult:
    """
    Validates, sanitizes, and deduplicates bounding boxes across all object instances.
    1. Validates coordinates [0, 1000], auto-repairs minor inversions, and discards zero/tiny area boxes (< 5 units).
    2. Performs conservative NMS deduplication to eliminate duplicate boxes covering the exact same physical space,
       while carefully preserving valid overlapping objects (overlapping people, stacked items).
    """
    all_instances = []

    # Step 1: Sanitize individual boxes
    for category in result.objects:
        cat_name = category.name.strip().lower()
        for instance in category.instances:
            bbox = instance.bounding_box
            if bbox is not None:
                sanitized = sanitize_box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, min_size_px_in_1000=5.0)
                if sanitized:
                    bbox.x_min = sanitized['x_min']
                    bbox.y_min = sanitized['y_min']
                    bbox.x_max = sanitized['x_max']
                    bbox.y_max = sanitized['y_max']
                    all_instances.append((cat_name, instance, {
                        'x_min': bbox.x_min,
                        'y_min': bbox.y_min,
                        'x_max': bbox.x_max,
                        'y_max': bbox.y_max
                    }))
                else:
                    logger.warning(f"Discarding tiny/zero-area bounding box for instance {instance.id}")
                    instance.bounding_box = None

    # Step 2: Overlap deduplication (IoU & Containment NMS)
    n = len(all_instances)
    discard_indices = set()

    for i in range(n):
        if i in discard_indices:
            continue
        cat_i, inst_i, box_i = all_instances[i]

        for j in range(i + 1, n):
            if j in discard_indices:
                continue
            cat_j, inst_j, box_j = all_instances[j]

            iou = calculate_iou(box_i, box_j)
            containment = calculate_containment(box_i, box_j)
            is_i_generic = cat_i in GENERIC_OBJECT_NAMES
            is_j_generic = cat_j in GENERIC_OBJECT_NAMES

            if cat_i == cat_j:
                # Same category: only drop near-identical duplicate boxes (IoU >= 0.85)
                # Avoid dropping legitimate overlapping items (e.g. 2 overlapping pens, overlapping people)
                if iou >= 0.85 or (containment >= 0.95 and iou >= 0.80):
                    discard_indices.add(j)
                    logger.info(f"Dedup: dropping identical duplicate '{inst_j.id}' of '{inst_i.id}' (IoU={iou:.2f})")
            elif iou >= 0.60 and (is_i_generic != is_j_generic):
                # Different names, same physical space, one is generic ("round object"): keep specific one
                loser = i if is_i_generic else j
                discard_indices.add(loser)
                logger.info(f"Dedup: dropping generic box in favor of specific one (IoU={iou:.2f})")
                if loser == i:
                    break

    # Apply discards
    for idx in discard_indices:
        _, inst_to_discard, _ = all_instances[idx]
        inst_to_discard.bounding_box = None

    return result

# ---------------------------------------------------------------------------
# DEDICATED LOCALIZATION PASS
# ---------------------------------------------------------------------------

class LocalizedBox(BaseModel):
    label: str = Field(description="Object category label, exactly one of the requested categories.")
    box_2d: List[int] = Field(description="[ymin, xmin, ymax, xmax] normalized to 0-1000 (Y FIRST).")

LOCALIZATION_TIMEOUT = 30.0

def localize_objects(
    image_part: types.Part,
    api_key: str,
    categories: List[Tuple[str, int]],
    model_name: str
) -> List[dict]:
    """
    Focused localization pass returning candidate detections with canonical coordinates.
    Gemini native box_2d is [ymin, xmin, ymax, xmax] (0-1000 grid).
    Converted explicitly to canonical {'x_min','y_min','x_max','y_max'}.
    """
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(LOCALIZATION_TIMEOUT * 1000))
    )
    wanted = ", ".join(f"{name} (about {cnt})" for name, cnt in categories)
    prompt = (
        f"Detect every visible instance of these objects: {wanted}.\n"
        "Return a JSON list. Each item: {\"label\": <one of the requested names>, "
        "\"box_2d\": [ymin, xmin, ymax, xmax]} with coordinates normalized to 0-1000.\n"
        "Rules: one tight box per physical object, hugging its visible edges (no padding, no neighbours); "
        "never one box around a group; do not box the table/background; ymin/ymax are vertical, xmin/xmax horizontal."
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=List[LocalizedBox],
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    response = client.models.generate_content(model=model_name, contents=[image_part, prompt], config=config)
    raw = response.text if response and getattr(response, "text", None) else ""
    data = json.loads(clean_json_text(raw)) if raw else []
    if isinstance(data, dict):
        data = data.get("objects") or data.get("detections") or []

    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        raw_box_2d = item.get("box_2d") or item.get("box2d")
        box = parse_gemini_box_2d(raw_box_2d)
        label = str(item.get("label") or "").strip().lower()
        if box and label and validate_bbox_coords(box['x_min'], box['y_min'], box['x_max'], box['y_max']):
            out.append({"label": label, "box": box})
    return out

def refine_bounding_boxes(
    result: GroundedAnalysisResult,
    image_part: types.Part,
    api_key: str,
    model_name: str
) -> GroundedAnalysisResult:
    """
    Refines bounding boxes from the main analysis using candidate boxes from the dedicated localization pass.
    
    Robust Architecture:
    1. Localization is a REFINEMENT mechanism, NEVER the sole source of truth.
    2. Objects are matched GEOMETRICALLY (IoU + Center Distance) within the same category.
    3. Consistency check: Candidate is accepted only if geometrically consistent with the original instance.
    4. FALLBACK RULE: Never delete a valid original box because refinement failed or returned fewer boxes!
    """
    categories = [(c.name.strip().lower(), max(len(c.instances), c.confirmed_count)) for c in result.objects if c.instances]
    if not categories:
        return result

    try:
        t0 = time.perf_counter()
        detections = localize_objects(image_part, api_key, categories, model_name)
        logger.info(f"[LOCALIZE] Retrieved {len(detections)} candidate boxes in {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"[LOCALIZE_FAILED] Keeping original analysis boxes due to exception: {e}")
        return result

    if not detections:
        logger.info("[LOCALIZE] No candidate detections returned; preserving all original boxes.")
        return result

    # Conservative deduplication among candidate detections (only drop near-identical IoU >= 0.85)
    kept_detections: List[dict] = []
    for d in sorted(detections, key=lambda d: (d["box"]["y_max"] - d["box"]["y_min"]) * (d["box"]["x_max"] - d["box"]["x_min"])):
        if any(labels_match(k["label"], d["label"]) and calculate_iou(k["box"], d["box"]) >= 0.85 for k in kept_detections):
            continue
        kept_detections.append(d)

    used_detection_indices = set()

    for cat in result.objects:
        cat_name = cat.name.strip().lower()

        # Find available candidate detections for this category
        candidate_indices = [
            i for i, d in enumerate(kept_detections)
            if i not in used_detection_indices and labels_match(d["label"], cat_name)
        ]

        if not candidate_indices:
            logger.info(f"[BBOX] No localization candidates for category='{cat_name}'; keeping original boxes.")
            continue

        # Separate instances with existing valid boxes vs unboxed instances
        boxed_instances = []
        unboxed_instances = []
        for inst_idx, inst in enumerate(cat.instances):
            if inst.bounding_box is not None:
                orig_b = {
                    'x_min': inst.bounding_box.x_min,
                    'y_min': inst.bounding_box.y_min,
                    'x_max': inst.bounding_box.x_max,
                    'y_max': inst.bounding_box.y_max
                }
                if validate_bbox_coords(orig_b['x_min'], orig_b['y_min'], orig_b['x_max'], orig_b['y_max']):
                    boxed_instances.append((inst_idx, inst, orig_b))
                else:
                    unboxed_instances.append((inst_idx, inst))
            else:
                unboxed_instances.append((inst_idx, inst))

        # Geometric matching for boxed instances (IoU + Center distance)
        # Compute match scores for all (instance, candidate) pairs
        match_candidates = []
        for b_idx, (orig_pos, inst, orig_b) in enumerate(boxed_instances):
            for c_idx in candidate_indices:
                cand_b = kept_detections[c_idx]["box"]
                iou = calculate_iou(orig_b, cand_b)
                center_dist = calculate_center_distance(orig_b, cand_b)

                # Consistency check:
                # 1. Significant IoU overlap (>= 0.20), OR
                # 2. Close center proximity (<= 200px) with at least minor overlap (IoU >= 0.05), OR
                # 3. Very close center proximity (<= 120px)
                is_consistent = (
                    iou >= 0.20 or
                    (center_dist <= 200.0 and iou >= 0.05) or
                    (center_dist <= 120.0)
                )

                if is_consistent:
                    # Match score: prioritize higher IoU and closer center distance
                    score = (iou * 1000.0) - center_dist
                    match_candidates.append({
                        'b_idx': b_idx,
                        'c_idx': c_idx,
                        'score': score,
                        'iou': iou,
                        'center_dist': center_dist,
                        'orig_pos': orig_pos,
                        'inst': inst,
                        'orig_b': orig_b,
                        'cand_b': cand_b
                    })

        # Sort candidate matches by score descending
        match_candidates.sort(key=lambda m: m['score'], reverse=True)

        assigned_boxed = set()
        assigned_candidates = set()

        for match in match_candidates:
            b_idx = match['b_idx']
            c_idx = match['c_idx']
            if b_idx in assigned_boxed or c_idx in assigned_candidates:
                continue

            # Accept refinement!
            assigned_boxed.add(b_idx)
            assigned_candidates.add(c_idx)
            used_detection_indices.add(c_idx)

            inst = match['inst']
            cand_b = match['cand_b']
            inst.bounding_box = BoundingBox(**cand_b)

            logger.info(
                f"[BBOX] category={cat_name} id={inst.id} "
                f"original=[{match['orig_b']['x_min']},{match['orig_b']['y_min']},{match['orig_b']['x_max']},{match['orig_b']['y_max']}] "
                f"localized=[{cand_b['x_min']},{cand_b['y_min']},{cand_b['x_max']},{cand_b['y_max']}] "
                f"IoU={match['iou']:.3f} center_distance={match['center_dist']:.1f} decision=ACCEPT"
            )

        # Log preservation of unmatched original boxes
        for b_idx, (orig_pos, inst, orig_b) in enumerate(boxed_instances):
            if b_idx not in assigned_boxed:
                logger.info(
                    f"[BBOX] category={cat_name} id={inst.id} "
                    f"original=[{orig_b['x_min']},{orig_b['y_min']},{orig_b['x_max']},{orig_b['y_max']}] "
                    f"decision=KEEP_ORIGINAL (no consistent candidate or candidate claimed by closer instance)"
                )

        # For unboxed instances (originally None), assign remaining unused candidates if available
        remaining_candidates = [c for c in candidate_indices if c not in assigned_candidates and c not in used_detection_indices]
        for (u_pos, inst) in unboxed_instances:
            if remaining_candidates:
                c_idx = remaining_candidates.pop(0)
                used_detection_indices.add(c_idx)
                cand_b = kept_detections[c_idx]["box"]
                inst.bounding_box = BoundingBox(**cand_b)
                logger.info(
                    f"[BBOX] category={cat_name} id={inst.id} original=None "
                    f"localized=[{cand_b['x_min']},{cand_b['y_min']},{cand_b['x_max']},{cand_b['y_max']}] decision=ASSIGN_NEW"
                )

    return result


def analyze_image_grounded(image: Union[Image.Image, types.Part], api_key: str, image_width: int = 0, image_height: int = 0) -> GroundedAnalysisResult:
    """
    Sends image or pre-encoded Part to Gemini VLM with structured Pydantic schema (GroundedAnalysisResult).
    Uses PRIMARY_VLM_MODEL and FALLBACK_VLM_MODEL with explicit timeouts, normalization, and field-level validation logging.
    
    Args:
        image: PIL Image or pre-encoded Part object.
        api_key: Gemini API key.
        image_width: Width in pixels of the image being analyzed (for spatial calibration).
        image_height: Height in pixels of the image being analyzed (for spatial calibration).
    """
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("API_KEY_ERROR: Gemini API key (VLM_API_KEY) is missing or unconfigured.")

    if isinstance(image, types.Part):
        image_part = image
    elif isinstance(image, Image.Image):
        img_bytes, mime_type, enc_w, enc_h = encode_vlm_image_part(image)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
        # Use encoded dimensions if caller didn't provide explicit dimensions
        if image_width <= 0 or image_height <= 0:
            image_width, image_height = enc_w, enc_h
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

    # Build dimension context string for user prompt
    dimension_context = ""
    if image_width > 0 and image_height > 0:
        aspect = "landscape" if image_width > image_height else ("portrait" if image_height > image_width else "square")
        dimension_context = f"\n\nIMAGE DIMENSIONS: This image is {image_width} × {image_height} pixels ({aspect}). Use this to calibrate your bounding box coordinate accuracy. The 0-1000 coordinate scale maps linearly across the full {image_width}px width (X axis) and {image_height}px height (Y axis)."

    last_exception = None
    t0 = time.perf_counter()

    for model_name in models_to_try:
        for attempt in range(1, 2 + 1):  # Max 2 attempts per model
            try:
                # On retry after MAX_TOKENS, use a hyper-concise user prompt instruction
                if attempt == 1:
                    user_prompt = f"Perform strict visual verification, physical instance counting, independent attribute analysis, TIGHT bounding box localization, and scene classification on this image. Return bounding_box coordinates as dict {{x_min, y_min, x_max, y_max}} in 0-1000 scale. Each box MUST tightly fit the visible object — no extra padding, no background space. Verify x_min < x_max and y_min < y_max for every box.{dimension_context}"
                else:
                    user_prompt = f"CONCISE RETRY: Previous response hit token limit. Return the exact GroundedAnalysisResult JSON schema using EXTREMELY concise 1-2 word attribute values, limit object categories to top 10 most prominent items, and keep scene summary under 15 words. Do NOT list trivial background micro-objects or write prose descriptions inside attributes.{dimension_context}"

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
                final_res = refine_bounding_boxes(final_res, image_part, api_key, model_name)
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
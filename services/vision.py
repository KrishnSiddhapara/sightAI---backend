import json
import logging
import time
import socket
import re
import io
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
from utils.config import (
    MAX_OUTPUT_TOKENS, PRIMARY_VLM_MODEL, FALLBACK_VLM_MODEL, ANALYSIS_TIMEOUT,
    BBOX_REFINE_ENABLED, BBOX_DETECTION_MODEL, BBOX_THINKING_BUDGET,
    BBOX_USE_STRICT_SCHEMA, BBOX_TARGETED_REDETECT, BBOX_CROP_REFINEMENT,
    LOCALIZATION_TIMEOUT
)
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
- OBJECT CATEGORY CONTROL: Exhaustively detect and localize ALL visible physical objects and items present in the scene without arbitrary truncation or omission. Do NOT cap or limit the number of valid items. (Only ignore trivial background micro-textures like individual leaves or tiny specks).
- ACCURATE OBJECT CATEGORY DISAMBIGUATION:
  * Organic vs Synthetic: A 'plant' MUST have real organic leaves, foliage, stems, or soil. Do NOT label smooth synthetic/plastic/metal containers, jars, pucks, or lids as 'plant' simply because they are green! A round jar/puck is a 'container' or 'round container'.
  * Light / Candle vs Figurine: A candle or tea light with a wax body, wick, or flame is a 'candle', NEVER an 'elephant figurine' or statue.
  * Sculptures / Figurines: Each elephant figurine must be a carved animal figure with ears/trunk/tusks. Never count a candle or empty space as an elephant figurine.
  * One Box Per Unique Item: Each distinct item gets exactly ONE bounding box. Never place multiple boxes on the same figurine or overlap adjacent figurines.
- WHOLE PHYSICAL OBJECTS ONLY: Detect complete, whole physical objects. Do NOT break down a composite object into its sub-components (e.g. do NOT detect 'cord', 'wire', or 'coil' of headphones separately from the headphones; do NOT detect 'keys' separately from the keyboard; do NOT detect 'trackpad' or 'screen' separately from the laptop). Always detect the whole physical entity!
- NO ARTIFACT OR EMPTY SPACE DETECTIONS: Never create bounding boxes on empty desk/table space, shadows, gaps between objects, or detached cords. Every box MUST tightly frame a real visible item.

STEP 2: PHYSICAL INSTANCE COUNTING & DEDUPLICATION
- Count distinct physical instances.
- DO NOT double-count an object under different names (e.g. if an item is an 'elephant figurine', do NOT list it as 'elephant figurine' AND also as 'round object' or 'candle').
- Standardize object category names to simple lowercase terms (e.g. 'person', 'car', 'dog', 'chair', 'bottle', 'bicycle', 'phone', 'elephant figurine', 'computer mouse', 'plant', 'towel', 'candle', 'container').
- For overlapping or partially hidden objects:
  * Count ONLY clearly distinguishable physical instances under 'confirmed_count'.
  * If an object is partially visible but cannot be confirmed, increment 'uncertain_count' and explain the reason in 'uncertainty_reason'. Do NOT guess or pad the confirmed count.

STEP 3: INDEPENDENT PERSON-BY-PERSON & INSTANCE ATTRIBUTE EXTRACTION
- EVERY person or object instance (person_1, elephant figurine_1, mouse_1, etc.) MUST be analyzed independently based strictly on their visible spatial region.
- NEVER create a generic description and copy it to multiple instances.
- ATTRIBUTES MUST BE CONCISE 1-4 WORD PHRASES (e.g. clothing='red t-shirt and jeans', pose='standing', action='waving', object_color='white marble'). NEVER write long prose, paragraphs, explanations, or commentary inside attribute fields!
- If an attribute (e.g. exact clothing, pose, action, age, brand) is not clearly visible, return "not clearly visible" or "unknown". DO NOT guess!

STEP 4: GROUNDED BOUNDING BOX LOCALIZATION & ACCURACY
- For EVERY confidently identified physical object instance, provide bounding box coordinates using 'box_2d': [ymin, xmin, ymax, xmax] normalized to 0 to 1000 scale:
  * ymin = Topmost vertical edge of the object (0 = top image boundary, 1000 = bottom image boundary)
  * xmin = Leftmost horizontal edge of the object (0 = left image boundary, 1000 = right image boundary)
  * ymax = Bottommost vertical edge of the object (0 to 1000, MUST be > ymin)
  * xmax = Rightmost horizontal edge of the object (0 to 1000, MUST be > xmin)
- CRITICAL BOUNDING BOX ACCURACY RULES:
  1. 100% BOUNDING BOX COVERAGE (ZERO UNBOXED INSTANCES): Every single confirmed object in 'instances' MUST have its own tight 'box_2d'. There must NEVER be an instance with 'box_2d': null or missing coordinates! If an object is visible, you MUST frame its exact visible extent.
  2. TIGHT VISUAL FIT: The bounding box MUST tightly hug the outermost visible physical pixels of the object. No extra margins, background padding, or shadows. Oversized or loose boxes are WRONG.
  3. EXACT OBJECT ANCHOR: The box MUST be placed directly on the visual center and extent of the object. Never shift a box onto empty space or onto a different neighboring object!
  4. ADJACENT OBJECTS (NO OVERLAP): Two distinct tabletop objects NEVER occupy the exact same physical space. If two objects sit next to each other, their boxes must stop at the visual separation line and NOT bleed into or overlap each other.
  5. NATIVE AXIS ALIGNMENT: ymin and ymax measure VERTICAL position (top to bottom); xmin and xmax measure HORIZONTAL position (left to right).
  6. INDEPENDENT INSTANCES: Each distinct physical instance MUST have its own separately calculated bounding box enclosing that exact item. NEVER copy, duplicate, or share coordinates between different objects.
  7. ONE BOX PER PHYSICAL ITEM: Never generate multiple overlapping boxes for the same physical item under different category names. Each real-world object gets exactly ONE bounding box.
  8. UNCERTAIN OBJECTS: If an item is too occluded, distant, or unclear to localize with a bounding box, do NOT place it in 'instances' or 'confirmed_count'. Instead, record it under 'uncertain_count' with an explanation in 'uncertainty_reason'.

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
          "box_2d": [150, 100, 800, 300],
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

from concurrent.futures import ThreadPoolExecutor, as_completed
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
    box_2d_to_dict,
    map_crop_box_to_full,
    apply_tile_nms,
    is_empty_background_crop
)

GENERIC_OBJECT_NAMES = {
    "round object", "small round object", "object", "thing", "item", "shape",
    "blob", "miscellaneous", "unknown", "coil", "cord", "cable", "wire", "part"
}

SURFACE_OBJECT_NAMES = {
    "towel", "white towel", "hand towel", "bath towel", "cloth", "table", "desk",
    "mat", "desk mat", "mousepad", "mouse pad", "tray", "plate", "dish", "bowl",
    "blanket", "bed", "floor", "rug", "carpet", "board", "shelf", "box", "basket", "napkin"
}

SUBPART_OBJECT_NAMES = {
    "coil", "cord", "cable", "wire", "part", "wheel", "key", "button",
    "screen", "strap", "handle", "plug", "dial", "switch"
}

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
    "eyeglasses": "glass",
    "sunglasses": "glass",
    "sunglass": "glass",
    "glasses": "glass",
    "spectacles": "glass",
    "computer mouse": "mouse",
    "laptop computer": "laptop",
    "notebook": "laptop",
    "computer keyboard": "keyboard",
    "wristwatch": "watch",
    "smartwatch": "watch",
    "watches": "watch",
    "elephant figurine": "elephant",
    "elephant statue": "elephant",
    "headphone": "headphones",
    "earphone": "headphones",
    "earphones": "headphones",
    "headset": "headphones",
    "usb": "adapter",
    "usb adapter": "adapter",
    "power adapter": "adapter",
    "wall adapter": "adapter",
    "charger": "adapter",
}

def normalize_label(label: str) -> str:
    """Normalizes object label for safe exact or alias comparison."""
    if not label:
        return ""
    cleaned = re.sub(r'[^a-z0-9\s]', '', label.lower().strip())
    if cleaned in SAFE_CATEGORY_ALIASES:
        return SAFE_CATEGORY_ALIASES[cleaned]

    tokens = cleaned.split()
    normalized_tokens = []
    for t in tokens:
        if t in SAFE_CATEGORY_ALIASES:
            normalized_tokens.append(SAFE_CATEGORY_ALIASES[t])
            continue
        if len(t) > 4 and t.endswith(('sses', 'shes', 'ches', 'xes')):
            stemmed = t[:-2]
            normalized_tokens.append(SAFE_CATEGORY_ALIASES.get(stemmed, stemmed))
        elif len(t) > 3 and t.endswith('s') and not t.endswith(('ss', 'us', 'is')):
            stemmed = t[:-1]
            normalized_tokens.append(SAFE_CATEGORY_ALIASES.get(stemmed, stemmed))
        else:
            normalized_tokens.append(t)
    base = " ".join(normalized_tokens)
    return SAFE_CATEGORY_ALIASES.get(base, base)

def labels_match(label1: str, label2: str) -> bool:
    """
    Checks if two category labels refer to the same semantic class.
    Uses exact normalized comparison, safe explicit alias mapping, and compound head-noun matching.
    Prevents false merges between distinct classes (e.g. 'cup' vs 'bottle', 'phone' vs 'laptop').
    """
    if not label1 or not label2:
        return False

    c1 = re.sub(r'[^a-z0-9\s]', '', label1.lower().strip())
    c2 = re.sub(r'[^a-z0-9\s]', '', label2.lower().strip())
    if not c1 or not c2:
        return False
    if c1 == c2:
        return True

    n1 = normalize_label(label1)
    n2 = normalize_label(label2)
    if n1 == n2:
        return True

    generic_heads = {"object", "item", "thing", "part", "shape", "blob"}

    for w1, w2 in [(n1.split(), n2.split()), (c1.split(), c2.split())]:
        if not w1 or not w2:
            continue
        if w1[-1] == w2[-1] and w1[-1] not in generic_heads:
            return True
        if len(w1) == 1 and w1[0] == w2[-1] and w1[0] not in generic_heads:
            return True
        if len(w2) == 1 and w2[0] == w1[-1] and w2[0] not in generic_heads:
            return True

    return False

def resolve_adjacent_box_overlaps(instances: List[Tuple[str, ObjectInstance, Dict[str, float]]]) -> None:
    """
    Resolves accidental boundary overlap between distinct adjacent physical objects.
    When two adjacent objects have a slight boundary overlap (IoU <= 0.15, overlap <= 12% of box size),
    adjusts their shared boundary to the midpoint so they cleanly touch without overlapping.
    Guaranteed to never significantly shift or shrink boxes away from the objects.
    """
    n = len(instances)
    for i in range(n):
        for j in range(i + 1, n):
            _, inst_i, _ = instances[i]
            _, inst_j, _ = instances[j]

            if inst_i.bounding_box is None or inst_j.bounding_box is None:
                continue

            bi = inst_i.bounding_box
            bj = inst_j.bounding_box

            # Current coordinates
            x_ov_min = max(bi.x_min, bj.x_min)
            x_ov_max = min(bi.x_max, bj.x_max)
            y_ov_min = max(bi.y_min, bj.y_min)
            y_ov_max = min(bi.y_max, bj.y_max)

            if x_ov_max <= x_ov_min or y_ov_max <= y_ov_min:
                continue

            ov_w = x_ov_max - x_ov_min
            ov_h = y_ov_max - y_ov_min

            box_i = {'x_min': bi.x_min, 'y_min': bi.y_min, 'x_max': bi.x_max, 'y_max': bi.y_max}
            box_j = {'x_min': bj.x_min, 'y_min': bj.y_min, 'x_max': bj.x_max, 'y_max': bj.y_max}
            iou = calculate_iou(box_i, box_j)
            containment = calculate_containment(box_i, box_j)

            # Only adjust minor border bleed (IoU <= 0.15 and containment < 0.35)
            if iou > 0.15 or containment >= 0.35:
                continue

            w_i = bi.x_max - bi.x_min
            w_j = bj.x_max - bj.x_min
            h_i = bi.y_max - bi.y_min
            h_j = bj.y_max - bj.y_min

            min_w = min(w_i, w_j)
            min_h = min(h_i, h_j)

            if ov_w < ov_h and min_w > 0 and (ov_w / min_w) <= 0.20:
                # Horizontally adjacent (minor side-by-side bleed)
                if (bi.x_min + bi.x_max) < (bj.x_min + bj.x_max):
                    # Box i is on left, Box j is on right
                    split_x = round((bi.x_max + bj.x_min) / 2.0, 1)
                    if (split_x - bi.x_min) >= 10.0 and (bj.x_max - split_x) >= 10.0:
                        bi.x_max = split_x
                        bj.x_min = split_x
                        logger.info(f"[ADJACENT_SEPARATION] Split X between {inst_i.id} and {inst_j.id} at x={split_x}")
                else:
                    # Box j is on left, Box i is on right
                    split_x = round((bj.x_max + bi.x_min) / 2.0, 1)
                    if (split_x - bj.x_min) >= 10.0 and (bi.x_max - split_x) >= 10.0:
                        bj.x_max = split_x
                        bi.x_min = split_x
                        logger.info(f"[ADJACENT_SEPARATION] Split X between {inst_j.id} and {inst_i.id} at x={split_x}")
            elif ov_h <= ov_w and min_h > 0 and (ov_h / min_h) <= 0.20:
                # Vertically adjacent (minor stacked bleed)
                if (bi.y_min + bi.y_max) < (bj.y_min + bj.y_max):
                    # Box i is on top, Box j is on bottom
                    split_y = round((bi.y_max + bj.y_min) / 2.0, 1)
                    if (split_y - bi.y_min) >= 10.0 and (bj.y_max - split_y) >= 10.0:
                        bi.y_max = split_y
                        bj.y_min = split_y
                        logger.info(f"[ADJACENT_SEPARATION] Split Y between {inst_i.id} and {inst_j.id} at y={split_y}")
                else:
                    # Box j is on top, Box i is on bottom
                    split_y = round((bj.y_max + bi.y_min) / 2.0, 1)
                    if (split_y - bj.y_min) >= 10.0 and (bi.y_max - split_y) >= 10.0:
                        bj.y_max = split_y
                        bi.y_min = split_y
                        logger.info(f"[ADJACENT_SEPARATION] Split Y between {inst_j.id} and {inst_i.id} at y={split_y}")


def sanitize_bounding_boxes(result: GroundedAnalysisResult) -> GroundedAnalysisResult:
    """
    Validates, sanitizes, deduplicates, and separates overlapping bounding boxes across all object instances.
    1. Validates coordinates [0, 1000], auto-repairs minor inversions, and discards zero/tiny area boxes (< 5 units).
    2. Performs NMS deduplication, sub-part suppression, and cross-category physical exclusion.
    3. Resolves accidental boundary overlaps between distinct adjacent physical objects.
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

    # Step 2: Overlap deduplication, sub-part suppression & cross-category physical exclusion
    n = len(all_instances)
    discard_indices = set()

    for i in range(n):
        if i in discard_indices:
            continue
        cat_i, inst_i, box_i = all_instances[i]
        area_i = (box_i['x_max'] - box_i['x_min']) * (box_i['y_max'] - box_i['y_min'])

        for j in range(i + 1, n):
            if j in discard_indices:
                continue
            cat_j, inst_j, box_j = all_instances[j]
            area_j = (box_j['x_max'] - box_j['x_min']) * (box_j['y_max'] - box_j['y_min'])

            iou = calculate_iou(box_i, box_j)
            containment = calculate_containment(box_i, box_j)
            center_dist = calculate_center_distance(box_i, box_j)
            is_i_generic = cat_i in GENERIC_OBJECT_NAMES
            is_j_generic = cat_j in GENERIC_OBJECT_NAMES
            is_alias_match = labels_match(cat_i, cat_j)
            same_category = (cat_i == cat_j)

            # Rule A: Same or alias categories deduplication
            # High IoU (>= 0.60) or near-coincident centers with IoU >= 0.40
            if (same_category or is_alias_match) and (
                iou >= 0.60 or (center_dist <= 25.0 and iou >= 0.40)
            ):
                loser = j if area_i >= area_j else i
                discard_indices.add(loser)
                logger.info(f"Dedup: dropping same-category duplicate between '{inst_i.id}' and '{inst_j.id}' (IoU={iou:.2f}, center_dist={center_dist:.1f})")
                if loser == i:
                    break

            # Rule B: One is a generic label / sub-part and significantly overlaps (IoU >= 0.50 or heavy containment in non-surface parent)
            elif (is_i_generic != is_j_generic):
                loser = i if is_i_generic else j
                is_loser_subpart = (all_instances[loser][0] in SUBPART_OBJECT_NAMES) or (all_instances[loser][0] in GENERIC_OBJECT_NAMES)
                winner_cat = all_instances[j if loser == i else i][0]
                is_winner_non_surface = winner_cat not in SURFACE_OBJECT_NAMES

                if iou >= 0.50 or (is_loser_subpart and is_winner_non_surface and containment >= 0.70):
                    discard_indices.add(loser)
                    logger.info(f"Dedup: dropping generic/subpart box '{all_instances[loser][1].id}' in favor of '{all_instances[j if loser==i else i][1].id}' (IoU={iou:.2f}, containment={containment:.2f})")
                    if loser == i:
                        break

            # Rule C: Cross-Category Identical Duplicate (two different categories on the EXACT same physical item, IoU >= 0.75)
            # NEVER applies if either category is a supporting surface (e.g. towel, mat, table - objects sit on surfaces!)
            elif (
                not same_category
                and not is_alias_match
                and cat_i not in SURFACE_OBJECT_NAMES
                and cat_j not in SURFACE_OBJECT_NAMES
                and iou >= 0.75
            ):
                count_i = sum(1 for c, _, _ in all_instances if c == cat_i)
                count_j = sum(1 for c, _, _ in all_instances if c == cat_j)
                if is_i_generic != is_j_generic:
                    loser = i if is_i_generic else j
                elif count_i != count_j:
                    loser = i if count_i > count_j else j
                else:
                    loser = i if area_i > area_j else j
                discard_indices.add(loser)
                logger.info(
                    f"Dedup: dropping duplicate cross-category label '{all_instances[loser][1].id}' ({all_instances[loser][0]}) "
                    f"colliding with '{all_instances[j if loser==i else i][1].id}' ({all_instances[j if loser==i else i][0]}) "
                    f"(IoU={iou:.2f})"
                )
                if loser == i:
                    break

    # Apply discards
    for idx in discard_indices:
        _, inst_to_discard, _ = all_instances[idx]
        inst_to_discard.bounding_box = None
        inst_to_discard.localization_status = "missing"

    # Step 3: Adjacent Box Boundary Separation (minor border bleeds only)
    active_instances = [all_instances[i] for i in range(n) if i not in discard_indices]
    resolve_adjacent_box_overlaps(active_instances)

    return result


def reconcile_unboxed_instances(result: GroundedAnalysisResult) -> GroundedAnalysisResult:
    """
    Ensures consistency across detected objects and instances:
    1. Verifies that confirmed_count accurately reflects detected instances.
    2. If confirmed_count is missing or smaller than boxed instances, synchronizes it.
    3. Sets localization_status ('ok' or 'missing') on each instance.
    """
    for cat in result.objects:
        boxed_count = sum(1 for inst in cat.instances if inst.bounding_box is not None)
        if cat.confirmed_count < boxed_count:
            cat.confirmed_count = boxed_count
        elif not cat.confirmed_count and cat.instances:
            cat.confirmed_count = len(cat.instances)

        for inst in cat.instances:
            if inst.bounding_box is not None:
                inst.localization_status = "ok"
            else:
                inst.localization_status = "missing"
                if not inst.uncertainty_reason:
                    inst.uncertainty_reason = "Object confirmed visually; tight bounding box could not be localized."

    return result

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# DEDICATED LOCALIZATION PASS
# ---------------------------------------------------------------------------

class LocalizedBox(BaseModel):
    label: str = Field(description="Object category label.")
    box_2d: List[int] = Field(description="[ymin, xmin, ymax, xmax] normalized to 0-1000 (Y FIRST, X SECOND).")

class DetectionEnvelope(BaseModel):
    detections: List[LocalizedBox] = Field(default_factory=list, description="List of localized physical objects.")


def localize_objects(
    image_part: types.Part,
    api_key: str,
    categories: List[Tuple[str, int]],
    model_name: str = BBOX_DETECTION_MODEL
) -> List[dict]:
    """
    Focused localization pass returning candidate detections with canonical coordinates.
    Uses Google's recommended 0-1000 [ymin, xmin, ymax, xmax] pattern inside a DetectionEnvelope.
    """
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(LOCALIZATION_TIMEOUT * 1000))
    )
    wanted = ", ".join(f"{name} (about {cnt})" for name, cnt in categories)
    prompt = (
        f"Detect every visible physical instance of these objects in the scene: {wanted}.\n"
        "Return a JSON object with 'detections': list of objects, each with 'label' and 'box_2d' [ymin, xmin, ymax, xmax] (0-1000 scale, Y FIRST, X SECOND).\n"
        "CRITICAL BOUNDING BOX RULES:\n"
        "1. TIGHT VISUAL FIT: Box MUST hug the outermost visible physical pixels of the object. Zero extra padding, zero desk/background space, zero shadows.\n"
        "2. NO ADJACENT OVERLAPPING: When two objects sit next to each other, their bounding boxes MUST NOT overlap or bleed into each other. Stop at the exact boundary separating them.\n"
        "3. WHOLE PHYSICAL OBJECTS: Box the entire object, never detached cords, sub-parts, or groups. Exactly one tight box per individual object."
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DetectionEnvelope,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=BBOX_THINKING_BUDGET),
    )
    response = client.models.generate_content(model=model_name, contents=[image_part, prompt], config=config)
    raw = response.text if response and getattr(response, "text", None) else ""
    data = json.loads(clean_json_text(raw)) if raw else {}
    if isinstance(data, dict):
        raw_items = data.get("detections") or data.get("objects") or []
    elif isinstance(data, list):
        raw_items = data
    else:
        raw_items = []

    out = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_box_2d = item.get("box_2d") or item.get("box2d")
        box = parse_gemini_box_2d(raw_box_2d)
        label = str(item.get("label") or "").strip().lower()
        if box and label and validate_bbox_coords(box['x_min'], box['y_min'], box['x_max'], box['y_max']):
            out.append({"label": label, "box": box})
    return out


def redetect_category(
    image_part: types.Part,
    api_key: str,
    cat_name: str,
    expected_count: int,
    model_name: str = BBOX_DETECTION_MODEL
) -> List[dict]:
    """
    Targeted re-detection for a specific category when main localization missed instances.
    """
    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(LOCALIZATION_TIMEOUT * 1000))
    )
    prompt = (
        f"Find and localize all {expected_count} visible physical instances of '{cat_name}' in this image.\n"
        "Return a JSON object with 'detections': list of objects with 'label' and 'box_2d': [ymin, xmin, ymax, xmax] (0-1000 scale).\n"
        "Provide tight bounding boxes around every visible instance."
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DetectionEnvelope,
        temperature=0.0,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )
    try:
        response = client.models.generate_content(model=model_name, contents=[image_part, prompt], config=config)
        raw = response.text if response and getattr(response, "text", None) else ""
        data = json.loads(clean_json_text(raw)) if raw else {}
        raw_items = data.get("detections") if isinstance(data, dict) else (data if isinstance(data, list) else [])
        out = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            box = parse_gemini_box_2d(item.get("box_2d") or item.get("box2d"))
            if box and validate_bbox_coords(box['x_min'], box['y_min'], box['x_max'], box['y_max']):
                out.append({"label": cat_name, "box": box})
        return out
    except Exception as e:
        logger.warning(f"[REDETECT_FAILED] Targeted re-detection for '{cat_name}' failed: {e}")
        return []


def refine_single_box_crop(
    pil_img: Image.Image,
    coarse_box: dict,
    label: str,
    api_key: str,
    model_name: str = BBOX_DETECTION_MODEL
) -> Optional[dict]:
    """
    Crops original PIL image around coarse_box with 20% padding (min size 120px),
    sends crop to Gemini to detect the tight box of label inside the crop,
    and maps the crop box back to full image 0-1000 coordinates.
    Accepts refined box only if IoU(refined, coarse) > 0.30 or IoU > 0.20 with center_dist < 100.
    """
    if pil_img is None or not coarse_box:
        return None

    try:
        orig_w, orig_h = pil_img.size
        w_1000 = coarse_box['x_max'] - coarse_box['x_min']
        h_1000 = coarse_box['y_max'] - coarse_box['y_min']

        pad_x = max(w_1000 * 0.20, 25.0)
        pad_y = max(h_1000 * 0.20, 25.0)

        crop_xmin_1000 = max(0.0, coarse_box['x_min'] - pad_x)
        crop_ymin_1000 = max(0.0, coarse_box['y_min'] - pad_y)
        crop_xmax_1000 = min(1000.0, coarse_box['x_max'] + pad_x)
        crop_ymax_1000 = min(1000.0, coarse_box['y_max'] + pad_y)

        px_x1 = max(0, int(round(crop_xmin_1000 * orig_w / 1000.0)))
        px_y1 = max(0, int(round(crop_ymin_1000 * orig_h / 1000.0)))
        px_x2 = min(orig_w, int(round(crop_xmax_1000 * orig_w / 1000.0)))
        px_y2 = min(orig_h, int(round(crop_ymax_1000 * orig_h / 1000.0)))

        # Ensure minimum crop dimension of 120px for visual context
        if (px_x2 - px_x1) < 120:
            diff = 120 - (px_x2 - px_x1)
            px_x1 = max(0, px_x1 - diff // 2)
            px_x2 = min(orig_w, px_x2 + (diff - diff // 2))
        if (px_y2 - px_y1) < 120:
            diff = 120 - (px_y2 - px_y1)
            px_y1 = max(0, px_y1 - diff // 2)
            px_y2 = min(orig_h, px_y2 + (diff - diff // 2))

        crop_pil = pil_img.crop((px_x1, px_y1, px_x2, px_y2))

        actual_crop_bounds_1000 = {
            'x_min': (px_x1 / orig_w) * 1000.0,
            'y_min': (px_y1 / orig_h) * 1000.0,
            'x_max': (px_x2 / orig_w) * 1000.0,
            'y_max': (px_y2 / orig_h) * 1000.0
        }

        crop_bytes, mime_type, _, _ = encode_vlm_image_part(crop_pil, max_dim=1024)
        crop_part = types.Part.from_bytes(data=crop_bytes, mime_type=mime_type)

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=10000)
        )
        prompt = (
            f"Locate the single tightest bounding box framing the '{label}' inside this cropped image region.\n"
            "Return a JSON object with 'box_2d': [ymin, xmin, ymax, xmax] (0-1000 scale relative to this crop).\n"
            "The box MUST tightly hug the physical edges of the item without extra margins or background."
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=LocalizedBox,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

        response = client.models.generate_content(model=model_name, contents=[crop_part, prompt], config=config)
        raw = response.text if response and getattr(response, "text", None) else ""
        data = json.loads(clean_json_text(raw)) if raw else {}

        raw_box_2d = data.get("box_2d") or data.get("box2d")
        if not raw_box_2d and isinstance(data, list) and data:
            raw_box_2d = data[0].get("box_2d") if isinstance(data[0], dict) else None

        crop_box = parse_gemini_box_2d(raw_box_2d)
        if not crop_box:
            return None

        full_mapped_box = map_crop_box_to_full(crop_box, actual_crop_bounds_1000)
        if not full_mapped_box:
            return None

        iou = calculate_iou(full_mapped_box, coarse_box)
        containment = calculate_containment(full_mapped_box, coarse_box)
        dist = calculate_center_distance(full_mapped_box, coarse_box)

        # Accept refined box if IoU >= 0.30 or if contained inside loose coarse box (containment >= 0.60 & dist <= 160)
        is_accepted = (
            iou >= 0.30 or
            (containment >= 0.60 and dist <= 160.0) or
            (iou >= 0.15 and dist <= 100.0)
        )

        if is_accepted:
            logger.info(f"[CROP_REFINE_ACCEPT] label='{label}' coarse={coarse_box} -> refined={full_mapped_box} (IoU={iou:.3f}, containment={containment:.3f}, dist={dist:.1f})")
            return full_mapped_box
        else:
            logger.info(f"[CROP_REFINE_REJECT] label='{label}' IoU={iou:.3f}, containment={containment:.3f} too low; keeping coarse.")
            return None

    except Exception as e:
        logger.warning(f"[CROP_REFINE_ERROR] Exception during crop refinement for '{label}': {e}")
        return None


def detect_objects_in_tiles(
    pil_img: Image.Image,
    cat_name: str,
    api_key: str,
    model_name: str = BBOX_DETECTION_MODEL
) -> List[dict]:
    """
    Splits pil_img into 2x2 overlapping tiles (25% overlap), detects cat_name in each tile,
    maps candidate boxes back to full image 0-1000, and applies tile NMS (IoU ~ 0.50).
    """
    if pil_img is None:
        return []

    orig_w, orig_h = pil_img.size
    tiles_1000 = [
        {"x_min": 0.0, "y_min": 0.0, "x_max": 625.0, "y_max": 625.0},
        {"x_min": 375.0, "y_min": 0.0, "x_max": 1000.0, "y_max": 625.0},
        {"x_min": 0.0, "y_min": 375.0, "x_max": 625.0, "y_max": 1000.0},
        {"x_min": 375.0, "y_min": 375.0, "x_max": 1000.0, "y_max": 1000.0},
    ]

    all_tile_dets = []

    def process_tile(tile_bounds: dict) -> List[dict]:
        px_x1 = int(round(tile_bounds["x_min"] * orig_w / 1000.0))
        px_y1 = int(round(tile_bounds["y_min"] * orig_h / 1000.0))
        px_x2 = int(round(tile_bounds["x_max"] * orig_w / 1000.0))
        px_y2 = int(round(tile_bounds["y_max"] * orig_h / 1000.0))

        tile_pil = pil_img.crop((px_x1, px_y1, px_x2, px_y2))
        tile_bytes, mime_type, _, _ = encode_vlm_image_part(tile_pil, max_dim=1024)
        tile_part = types.Part.from_bytes(data=tile_bytes, mime_type=mime_type)

        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=10000)
        )
        prompt = (
            f"Detect all visible physical instances of '{cat_name}' in this image tile.\n"
            "Return a JSON object with 'detections': list of objects with 'label' and 'box_2d': [ymin, xmin, ymax, xmax] (0-1000 scale)."
        )
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=DetectionEnvelope,
            temperature=0.0,
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

        try:
            resp = client.models.generate_content(model=model_name, contents=[tile_part, prompt], config=config)
            raw = resp.text if resp and getattr(resp, "text", None) else ""
            data = json.loads(clean_json_text(raw)) if raw else {}
            items = data.get("detections") if isinstance(data, dict) else (data if isinstance(data, list) else [])
            tile_mapped = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                cb = parse_gemini_box_2d(item.get("box_2d") or item.get("box2d"))
                if cb:
                    mapped = map_crop_box_to_full(cb, tile_bounds)
                    if mapped:
                        tile_mapped.append({"label": cat_name, "box": mapped})
            return tile_mapped
        except Exception as e:
            logger.warning(f"[TILE_DETECT_ERROR] Error in tile detection: {e}")
            return []

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_tile, tb) for tb in tiles_1000]
        for fut in as_completed(futures):
            res_dets = fut.result()
            if res_dets:
                all_tile_dets.extend(res_dets)

    final_tile_dets = apply_tile_nms(all_tile_dets, iou_threshold=0.50)
    logger.info(f"[TILE_DETECTION] Category '{cat_name}': {len(all_tile_dets)} raw tile detections -> {len(final_tile_dets)} after NMS")
    return final_tile_dets


def refine_bounding_boxes(
    result: GroundedAnalysisResult,
    image_part: types.Part,
    api_key: str,
    model_name: str,
    pil_image: Optional[Image.Image] = None
) -> GroundedAnalysisResult:
    """
    Refines bounding boxes from the main analysis using candidate boxes from the dedicated localization pass,
    crop-based spatial refinement, tile-based missing object detection, and background quality filtering.
    """
    if not BBOX_REFINE_ENABLED:
        logger.info("[BBOX] BBOX_REFINE_ENABLED is False; skipping dedicated localization pass.")
        for cat in result.objects:
            for inst in cat.instances:
                inst.localization_status = "ok" if inst.bounding_box is not None else "missing"
        return result

    categories = [(c.name.strip().lower(), max(len(c.instances), c.confirmed_count)) for c in result.objects if c.instances]
    if not categories:
        return result

    try:
        t0 = time.perf_counter()
        detections = localize_objects(image_part, api_key, categories, model_name=BBOX_DETECTION_MODEL)
        logger.info(f"[LOCALIZE] Retrieved {len(detections)} candidate boxes in {time.perf_counter() - t0:.2f}s")
    except Exception as e:
        logger.warning(f"[LOCALIZE_FAILED] Keeping original analysis boxes due to exception: {e}")
        for cat in result.objects:
            for inst in cat.instances:
                inst.localization_status = "ok" if inst.bounding_box is not None else "missing"
        return result

    if not detections:
        logger.info("[LOCALIZE] No candidate detections returned; preserving all original boxes.")
        for cat in result.objects:
            for inst in cat.instances:
                inst.localization_status = "ok" if inst.bounding_box is not None else "missing"
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

        total_inst_count = len(cat.instances)
        # Targeted re-detection & tile detection if verified instances lack candidate detections
        if len(candidate_indices) < total_inst_count and total_inst_count > 0:
            if BBOX_TARGETED_REDETECT:
                logger.info(f"[BBOX_REDETECT] Targeted re-detection for '{cat_name}' (expected: {total_inst_count}, candidates: {len(candidate_indices)})")
                redetected = redetect_category(image_part, api_key, cat_name, total_inst_count)
                for rd in redetected:
                    if not any(calculate_iou(rd["box"], k["box"]) >= 0.85 for k in kept_detections):
                        new_idx = len(kept_detections)
                        kept_detections.append(rd)
                        candidate_indices.append(new_idx)

            # If still short, run tile detection on overlapping tiles
            if len(candidate_indices) < total_inst_count and pil_image is not None:
                logger.info(f"[BBOX_TILE_REDETECT] Tile detection for '{cat_name}' (expected: {total_inst_count}, candidates: {len(candidate_indices)})")
                tile_dets = detect_objects_in_tiles(pil_image, cat_name, api_key)
                for td in tile_dets:
                    if not any(calculate_iou(td["box"], k["box"]) >= 0.85 for k in kept_detections):
                        new_idx = len(kept_detections)
                        kept_detections.append(td)
                        candidate_indices.append(new_idx)

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
        match_candidates = []
        for b_idx, (orig_pos, inst, orig_b) in enumerate(boxed_instances):
            for c_idx in candidate_indices:
                cand_b = kept_detections[c_idx]["box"]
                iou = calculate_iou(orig_b, cand_b)
                center_dist = calculate_center_distance(orig_b, cand_b)

                single_pair = (len(boxed_instances) == 1 and len(candidate_indices) == 1)
                is_consistent = (
                    iou >= 0.15 or
                    (center_dist <= 150.0 and iou >= 0.05) or
                    (center_dist <= 60.0) or
                    (single_pair and iou >= 0.03 and center_dist <= 250.0)
                )

                if is_consistent:
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

        match_candidates.sort(key=lambda m: m['score'], reverse=True)

        assigned_boxed = set()
        assigned_candidates = set()

        for match in match_candidates:
            b_idx = match['b_idx']
            c_idx = match['c_idx']
            if b_idx in assigned_boxed or c_idx in assigned_candidates:
                continue

            assigned_boxed.add(b_idx)
            assigned_candidates.add(c_idx)
            used_detection_indices.add(c_idx)

            inst = match['inst']
            cand_b = match['cand_b']
            inst.bounding_box = BoundingBox(**cand_b)
            inst.localization_status = "ok"

        for b_idx, (orig_pos, inst, orig_b) in enumerate(boxed_instances):
            if b_idx not in assigned_boxed:
                inst.localization_status = "ok"

        remaining_candidates = [c for c in candidate_indices if c not in assigned_candidates and c not in used_detection_indices]
        for (u_pos, inst) in unboxed_instances:
            if remaining_candidates:
                c_idx = remaining_candidates.pop(0)
                cand_b = kept_detections[c_idx]["box"]
                used_detection_indices.add(c_idx)
                inst.bounding_box = BoundingBox(**cand_b)
                inst.localization_status = "ok"
            else:
                inst.localization_status = "missing"

    # Stage 2: Bounded Concurrent Crop Refinement
    if BBOX_CROP_REFINEMENT and pil_image is not None:
        crop_refine_tasks = []
        for cat in result.objects:
            cat_name = cat.name.strip().lower()
            for inst in cat.instances:
                if inst.bounding_box is not None:
                    c_box = {
                        'x_min': inst.bounding_box.x_min,
                        'y_min': inst.bounding_box.y_min,
                        'x_max': inst.bounding_box.x_max,
                        'y_max': inst.bounding_box.y_max
                    }
                    crop_refine_tasks.append((cat_name, inst, c_box))

        if crop_refine_tasks:
            logger.info(f"[CROP_REFINE] Launching concurrent crop refinement for {len(crop_refine_tasks)} boxes...")
            with ThreadPoolExecutor(max_workers=6) as executor:
                future_map = {
                    executor.submit(refine_single_box_crop, pil_image, c_box, c_name, api_key): inst
                    for (c_name, inst, c_box) in crop_refine_tasks
                }
                for fut in as_completed(future_map):
                    target_inst = future_map[fut]
                    refined = fut.result()
                    if refined:
                        target_inst.bounding_box = BoundingBox(**refined)
                        target_inst.localization_status = "ok"

    # Stage 3: Empty Background Sanity Filtering
    if pil_image is not None:
        orig_w, orig_h = pil_image.size
        for cat in result.objects:
            for inst in cat.instances:
                if inst.bounding_box is not None:
                    b = inst.bounding_box
                    px_x1 = max(0, int(round(b.x_min * orig_w / 1000.0)))
                    px_y1 = max(0, int(round(b.y_min * orig_h / 1000.0)))
                    px_x2 = min(orig_w, int(round(b.x_max * orig_w / 1000.0)))
                    px_y2 = min(orig_h, int(round(b.y_max * orig_h / 1000.0)))
                    if (px_x2 - px_x1) >= 10 and (px_y2 - px_y1) >= 10:
                        crop_check = pil_image.crop((px_x1, px_y1, px_x2, px_y2))
                        if is_empty_background_crop(crop_check, stddev_threshold=6.0):
                            logger.warning(f"[EMPTY_BG_DROP] Dropping box for '{inst.id}' placed on empty background.")
                            inst.bounding_box = None
                            inst.localization_status = "missing"

    return result



def analyze_image_grounded(
    image: Union[Image.Image, types.Part],
    api_key: str,
    image_width: int = 0,
    image_height: int = 0,
    pil_image: Optional[Image.Image] = None
) -> GroundedAnalysisResult:
    """
    Sends image or pre-encoded Part to Gemini VLM with structured Pydantic schema (GroundedAnalysisResult).
    Uses PRIMARY_VLM_MODEL and FALLBACK_VLM_MODEL with explicit timeouts, normalization, and field-level validation logging.
    
    Args:
        image: PIL Image or pre-encoded Part object.
        api_key: Gemini API key.
        image_width: Width in pixels of the image being analyzed (for spatial calibration).
        image_height: Height in pixels of the image being analyzed (for spatial calibration).
        pil_image: Optional original PIL Image for crop refinement & tile detection.
    """
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("API_KEY_ERROR: Gemini API key (VLM_API_KEY) is missing or unconfigured.")

    pil_img = pil_image
    if isinstance(image, types.Part):
        image_part = image
        if pil_img is None and hasattr(image_part, "inline_data") and getattr(image_part.inline_data, "data", None):
            try:
                pil_img = Image.open(io.BytesIO(image_part.inline_data.data))
                pil_img.load()
            except Exception:
                pass
    elif isinstance(image, Image.Image):
        if pil_img is None:
            pil_img = image
        img_bytes, mime_type, enc_w, enc_h = encode_vlm_image_part(image)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
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

    last_exception = None
    t0 = time.perf_counter()

    for model_name in models_to_try:
        for attempt in range(1, 2 + 1):  # Max 2 attempts per model
            try:
                # On retry after MAX_TOKENS, use a hyper-concise user prompt instruction
                if attempt == 1:
                    user_prompt = "Perform strict visual verification, physical instance counting, independent attribute analysis, TIGHT bounding box localization, and scene classification on this image. Provide 'box_2d': [ymin, xmin, ymax, xmax] normalized to 0-1000 for each detected object instance (ymin/ymax vertical, xmin/xmax horizontal). Each box MUST tightly fit the visible object."
                else:
                    user_prompt = "CONCISE RETRY: Previous response hit token limit. Return the exact GroundedAnalysisResult JSON schema using EXTREMELY concise 1-2 word attribute values, detect all visible physical items without omitting any, and keep scene summary under 15 words. Provide 'box_2d': [ymin, xmin, ymax, xmax] normalized to 0-1000 for instances."

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
                final_res = refine_bounding_boxes(final_res, image_part, api_key, model_name, pil_image=pil_img)
                final_res = sanitize_bounding_boxes(final_res)
                final_res = reconcile_unboxed_instances(final_res)
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
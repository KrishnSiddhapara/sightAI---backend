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
                sanitized = sanitize_box(bbox.x_min, bbox.y_min, bbox.x_max, bbox.y_max, min_size_px_in_1000=2.0)
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
            # High IoU (>= 0.40), high containment (>= 0.65), or near-coincident centers (center_dist <= 30px with IoU >= 0.20)
            if (same_category or is_alias_match) and (
                iou >= 0.40 or containment >= 0.65 or (center_dist <= 30.0 and iou >= 0.20)
            ):
                loser = j if area_i >= area_j else i
                discard_indices.add(loser)
                logger.info(f"Dedup: dropping duplicate between '{inst_i.id}' and '{inst_j.id}' (IoU={iou:.2f}, containment={containment:.2f}, center_dist={center_dist:.1f})")
                if loser == i:
                    break

            # Rule B: One is a generic name / sub-part and significantly overlaps (IoU >= 0.30 or containment >= 0.50)
            elif (is_i_generic != is_j_generic) and (iou >= 0.30 or containment >= 0.50):
                loser = i if is_i_generic else j
                discard_indices.add(loser)
                logger.info(f"Dedup: dropping generic box '{all_instances[loser][1].id}' in favor of specific item (IoU={iou:.2f})")
                if loser == i:
                    break

            # Rule C: Heavy nested containment (containment >= 0.75) where one box is a small sub-part of another (area ratio < 0.35)
            # ONLY suppresses if the smaller object is an actual sub-part (coil, cord, wire) or generic name,
            # and the larger object is NOT a surface/cloth (objects sit on towels/tables, they are not sub-parts of towels!)
            smaller_cat = cat_j if area_j < area_i else cat_i
            larger_cat = cat_i if area_j < area_i else cat_j
            is_smaller_subpart = (smaller_cat in SUBPART_OBJECT_NAMES) or (smaller_cat in GENERIC_OBJECT_NAMES)
            is_larger_surface = (larger_cat in SURFACE_OBJECT_NAMES)
            area_ratio = (min(area_i, area_j) / max(area_i, area_j)) if max(area_i, area_j) > 0 else 0

            if is_smaller_subpart and not is_larger_surface and containment >= 0.70 and area_ratio < 0.35:
                loser = j if area_j < area_i else i
                discard_indices.add(loser)
                logger.info(f"Dedup: dropping small nested sub-part '{all_instances[loser][1].id}' (containment={containment:.2f})")
                if loser == i:
                    break

            # Rule D: Cross-Category Physical Exclusion
            # Two distinct solid tabletop objects (e.g. candle vs elephant figurine) cannot occupy the same 3D space.
            # Applies ONLY when:
            # 1. Categories are DIFFERENT and NOT aliases.
            # 2. NEITHER category is a supporting surface (e.g. towel, mat, tray, table - objects sit on surfaces!).
            # 3. High mutual footprint overlap (IoU >= 0.35, or high containment >= 0.65 with comparable area ratio >= 0.40).
            elif (
                not same_category
                and not is_alias_match
                and (cat_i not in SURFACE_OBJECT_NAMES and cat_j not in SURFACE_OBJECT_NAMES)
                and (iou >= 0.35 or (containment >= 0.65 and area_ratio >= 0.40))
            ):
                count_i = sum(1 for c, _, _ in all_instances if c == cat_i)
                count_j = sum(1 for c, _, _ in all_instances if c == cat_j)

                if is_i_generic != is_j_generic:
                    loser = i if is_i_generic else j
                elif count_i != count_j:
                    # The multi-instance category (e.g. 5 elephant figurines) has a spurious extra box on the unique item (e.g. 1 candle)
                    loser = i if count_i > count_j else j
                else:
                    loser = i if area_i > area_j else j

                discard_indices.add(loser)
                logger.info(
                    f"Dedup: dropping cross-category collision '{all_instances[loser][1].id}' ({all_instances[loser][0]}) "
                    f"colliding with '{all_instances[j if loser==i else i][1].id}' ({all_instances[j if loser==i else i][0]}) "
                    f"(IoU={iou:.2f}, containment={containment:.2f})"
                )
                if loser == i:
                    break

    # Apply discards
    for idx in discard_indices:
        _, inst_to_discard, _ = all_instances[idx]
        inst_to_discard.bounding_box = None

    # Step 3: Adjacent Box Boundary Separation (minor border bleeds only)
    active_instances = [all_instances[i] for i in range(n) if i not in discard_indices]
    resolve_adjacent_box_overlaps(active_instances)

    return result


def reconcile_unboxed_instances(result: GroundedAnalysisResult) -> GroundedAnalysisResult:
    """
    Ensures consistency across detected objects and instances:
    1. Verifies that confirmed_count accurately reflects detected instances.
    2. If confirmed_count is missing or smaller than boxed instances, synchronizes it.
    3. Preserves all instance attributes and coordinates.
    """
    for cat in result.objects:
        boxed_count = sum(1 for inst in cat.instances if inst.bounding_box is not None)
        if cat.confirmed_count < boxed_count:
            cat.confirmed_count = boxed_count
        elif not cat.confirmed_count and cat.instances:
            cat.confirmed_count = len(cat.instances)

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
        f"Detect every visible physical instance of these objects: {wanted}.\n"
        "Return a JSON list. Each item: {\"label\": <one of the requested names>, "
        "\"box_2d\": [ymin, xmin, ymax, xmax]} with coordinates normalized to 0-1000.\n"
        "CRITICAL BOUNDING BOX RULES:\n"
        "1. TIGHT VISUAL FIT: Box MUST hug the outermost visible physical pixels of the object. Zero extra padding, zero desk/background space, zero shadows.\n"
        "2. NO ADJACENT OVERLAPPING: When two objects sit next to each other, their bounding boxes MUST NOT overlap or bleed into each other. Stop at the exact boundary separating them.\n"
        "3. WHOLE PHYSICAL OBJECTS: Box the entire object, never detached cords, sub-parts, or groups. Exactly one tight box per individual object."
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

                # Strict geometric consistency check:
                # Refinement candidate MUST be anchored to the original detected object:
                # 1. Clear geometric overlap: iou >= 0.15, OR
                # 2. Moderate overlap with close center: iou >= 0.05 and center_dist <= 150.0, OR
                # 3. Very close center proximity: center_dist <= 60.0 (where even tiny items coincide), OR
                # 4. Single-pair with minor overlap: single_pair and iou >= 0.03 and center_dist <= 250.0.
                # A candidate with 0 IoU and center_dist > 60px is REJECTED to prevent moving boxes to empty space!
                single_pair = (len(boxed_instances) == 1 and len(candidate_indices) == 1)
                is_consistent = (
                    iou >= 0.15 or
                    (center_dist <= 150.0 and iou >= 0.05) or
                    (center_dist <= 60.0) or
                    (single_pair and iou >= 0.03 and center_dist <= 250.0)
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
        # BUT ONLY if candidate does NOT collide with an already assigned box in the scene
        remaining_candidates = [c for c in candidate_indices if c not in assigned_candidates and c not in used_detection_indices]
        for (u_pos, inst) in unboxed_instances:
            while remaining_candidates:
                c_idx = remaining_candidates.pop(0)
                cand_b = kept_detections[c_idx]["box"]

                # Collision check against existing boxes across all categories
                collides_existing = False
                for other_cat in result.objects:
                    for other_inst in other_cat.instances:
                        if other_inst.bounding_box is not None:
                            ob = {
                                'x_min': other_inst.bounding_box.x_min,
                                'y_min': other_inst.bounding_box.y_min,
                                'x_max': other_inst.bounding_box.x_max,
                                'y_max': other_inst.bounding_box.y_max
                            }
                            other_is_surface = other_cat.name.lower() in SURFACE_OBJECT_NAMES
                            this_is_surface = cat_name in SURFACE_OBJECT_NAMES
                            cand_area = (cand_b['x_max'] - cand_b['x_min']) * (cand_b['y_max'] - cand_b['y_min'])
                            ob_area = (ob['x_max'] - ob['x_min']) * (ob['y_max'] - ob['y_min'])
                            rel_area_ratio = (min(cand_area, ob_area) / max(cand_area, ob_area)) if max(cand_area, ob_area) > 0 else 0

                            if other_is_surface or this_is_surface:
                                # Overlap with a supporting surface (e.g. smartwatch resting on towel) is normal unless identical
                                if calculate_iou(cand_b, ob) >= 0.75:
                                    collides_existing = True
                                    break
                            else:
                                if calculate_iou(cand_b, ob) >= 0.35 or (calculate_containment(cand_b, ob) >= 0.65 and rel_area_ratio >= 0.40):
                                    collides_existing = True
                                    break
                    if collides_existing:
                        break

                if not collides_existing:
                    used_detection_indices.add(c_idx)
                    inst.bounding_box = BoundingBox(**cand_b)
                    logger.info(
                        f"[BBOX] category={cat_name} id={inst.id} original=None "
                        f"localized=[{cand_b['x_min']},{cand_b['y_min']},{cand_b['x_max']},{cand_b['y_max']}] decision=ASSIGN_NEW"
                    )
                    break

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
                final_res = refine_bounding_boxes(final_res, image_part, api_key, model_name)
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
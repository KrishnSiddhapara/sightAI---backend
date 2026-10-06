"""
Centralized Bounding Box & Image Coordinate Transformation Utility for SightAI Backend.

Standard coordinate convention throughout SightAI:
All bounding boxes are represented canonically as:
{
    'x_min': float,  # Left horizontal boundary (0 - 1000 scale)
    'y_min': float,  # Top vertical boundary (0 - 1000 scale)
    'x_max': float,  # Right horizontal boundary (0 - 1000 scale)
    'y_max': float   # Bottom vertical boundary (0 - 1000 scale)
}

Gemini Native Convention:
Gemini native box_2d is strictly [ymin, xmin, ymax, xmax] on a 0-1000 scale.
This must ALWAYS be converted explicitly without heuristic order guessing:
    x_min = xmin
    y_min = ymin
    x_max = xmax
    y_max = ymax
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import math
import logging

logger = logging.getLogger(__name__)


def validate_bbox_coords(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    min_size_px_in_1000: float = 2.0
) -> bool:
    """
    Validates if coordinates form a geometrically valid, non-inverted bounding box
    within the [0, 1000] canonical range with at least min_size_px_in_1000 width/height.
    """
    for v in (x_min, y_min, x_max, y_max):
        if v is None or math.isnan(v) or math.isinf(v):
            return False
        if v < 0.0 or v > 1000.0:
            return False

    if x_max <= x_min or y_max <= y_min:
        return False

    if (x_max - x_min) < min_size_px_in_1000 or (y_max - y_min) < min_size_px_in_1000:
        return False

    return True


def sanitize_box(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    min_size_px_in_1000: float = 2.0
) -> Optional[Dict[str, float]]:
    """
    Validates, repairs minor inversions, clamps to [0, 1000],
    and discards boxes smaller than min_size_px_in_1000.
    """
    try:
        x1 = float(x_min)
        y1 = float(y_min)
        x2 = float(x_max)
        y2 = float(y_max)
    except (ValueError, TypeError):
        return None

    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return None

    xmin = max(0.0, min(1000.0, min(x1, x2)))
    xmax = max(0.0, min(1000.0, max(x1, x2)))
    ymin = max(0.0, min(1000.0, min(y1, y2)))
    ymax = max(0.0, min(1000.0, max(y1, y2)))

    if (xmax - xmin) < min_size_px_in_1000 or (ymax - ymin) < min_size_px_in_1000:
        return None

    return {
        'x_min': round(xmin, 2),
        'y_min': round(ymin, 2),
        'x_max': round(xmax, 2),
        'y_max': round(ymax, 2)
    }


def parse_gemini_box_2d(raw_box: Any) -> Optional[Dict[str, float]]:
    """
    Explicitly converts Gemini native box_2d / box2d format:
        [ymin, xmin, ymax, xmax] (normalized 0 - 1000 or 0.0 - 1.0)
    or dictionary {'ymin': ..., 'xmin': ..., 'ymax': ..., 'xmax': ...}
    into canonical:
        {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}

    IMPORTANT:
    Never infers coordinate order from mathematical validity!
    In Gemini native format, element 0 is ALWAYS ymin, element 1 is ALWAYS xmin,
    element 2 is ALWAYS ymax, element 3 is ALWAYS xmax.
    """
    if isinstance(raw_box, dict):
        ymin = raw_box.get('ymin') if raw_box.get('ymin') is not None else raw_box.get('y_min')
        xmin = raw_box.get('xmin') if raw_box.get('xmin') is not None else raw_box.get('x_min')
        ymax = raw_box.get('ymax') if raw_box.get('ymax') is not None else raw_box.get('y_max')
        xmax = raw_box.get('xmax') if raw_box.get('xmax') is not None else raw_box.get('x_max')
        if all(v is not None for v in (ymin, xmin, ymax, xmax)):
            raw_box = [ymin, xmin, ymax, xmax]

    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
        return None

    try:
        ymin, xmin, ymax, xmax = [float(v) for v in raw_box]
    except (ValueError, TypeError):
        return None

    if any(math.isnan(v) or math.isinf(v) for v in (ymin, xmin, ymax, xmax)):
        return None

    # Whole-box scale detection: if all non-zero values are <= 1.0, scale to 0-1000
    if all(0.0 <= v <= 1.0 for v in (ymin, xmin, ymax, xmax)):
        ymin *= 1000.0
        xmin *= 1000.0
        ymax *= 1000.0
        xmax *= 1000.0

    sanitized = sanitize_box(x_min=xmin, y_min=ymin, x_max=xmax, y_max=ymax, min_size_px_in_1000=2.0)
    if sanitized:
        logger.debug(f"[BBOX] coordinate_format=gemini_box_2d raw={raw_box} normalized_box={sanitized}")
    return sanitized


def parse_standard_bbox(raw_box: Any) -> Optional[Dict[str, float]]:
    """
    Explicitly converts standard bounding boxes into canonical format:
    {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float} (0-1000 scale)

    Handles:
    - Dictionary with explicit keys: x_min/xmin/xMin/left, y_min/ymin/yMin/top, x_max/xmax/xMax/right, y_max/ymax/yMax/bottom
    - 4-element list/tuple in standard order [x_min, y_min, x_max, y_max]

    IMPORTANT:
    Never infers coordinate order from mathematical validity!
    """
    if raw_box is None:
        return None

    x1, y1, x2, y2 = None, None, None, None

    # Case 1: Dictionary format with explicit keys
    if isinstance(raw_box, dict):
        raw_x_min = raw_box.get('x_min') if raw_box.get('x_min') is not None else raw_box.get('xmin', raw_box.get('xMin', raw_box.get('left')))
        raw_y_min = raw_box.get('y_min') if raw_box.get('y_min') is not None else raw_box.get('ymin', raw_box.get('yMin', raw_box.get('top')))
        raw_x_max = raw_box.get('x_max') if raw_box.get('x_max') is not None else raw_box.get('xmax', raw_box.get('xMax', raw_box.get('right')))
        raw_y_max = raw_box.get('y_max') if raw_box.get('y_max') is not None else raw_box.get('ymax', raw_box.get('yMax', raw_box.get('bottom')))

        if all(v is not None for v in [raw_x_min, raw_y_min, raw_x_max, raw_y_max]):
            try:
                x1, y1, x2, y2 = float(raw_x_min), float(raw_y_min), float(raw_x_max), float(raw_y_max)
            except (ValueError, TypeError):
                return None

    # Case 2: List or Tuple in standard [x_min, y_min, x_max, y_max] order
    elif isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        try:
            x1, y1, x2, y2 = [float(v) for v in raw_box]
        except (ValueError, TypeError):
            return None

    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    if any(math.isnan(v) or math.isinf(v) for v in (x1, y1, x2, y2)):
        return None

    # Whole-box scale detection: if all non-zero values are <= 1.0, scale to 0-1000
    if all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
        x1 *= 1000.0
        y1 *= 1000.0
        x2 *= 1000.0
        y2 *= 1000.0

    sanitized = sanitize_box(x_min=x1, y_min=y1, x_max=x2, y_max=y2, min_size_px_in_1000=2.0)
    if sanitized:
        logger.debug(f"[BBOX] coordinate_format=standard_bbox raw={raw_box} normalized_box={sanitized}")
    return sanitized


def parse_and_normalize_bbox(raw_box: Any, is_gemini_box_2d: bool = False) -> Optional[Dict[str, float]]:
    """
    Backwards-compatible wrapper.
    If is_gemini_box_2d is True, parses as Gemini box_2d [ymin, xmin, ymax, xmax].
    Otherwise parses as standard bounding box ({x_min, y_min, ...} or [x_min, y_min, x_max, y_max]).
    """
    if raw_box is None:
        return None

    if is_gemini_box_2d:
        return parse_gemini_box_2d(raw_box)

    return parse_standard_bbox(raw_box)


def box_2d_to_dict(raw_box: Any) -> Optional[Dict[str, float]]:
    """
    Backwards-compatible alias for parse_gemini_box_2d.
    Converts Gemini native box_2d format [ymin, xmin, ymax, xmax] (0-1000)
    to canonical {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float}.
    """
    return parse_gemini_box_2d(raw_box)


def calculate_box_center(box: Dict[str, float]) -> Tuple[float, float]:
    """
    Returns center (center_x, center_y) of a bounding box in 0-1000 space.
    """
    cx = (box['x_min'] + box['x_max']) / 2.0
    cy = (box['y_min'] + box['y_max']) / 2.0
    return cx, cy


def calculate_center_distance(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculates Euclidean distance between centers of two 0-1000 scale bounding boxes.
    """
    if not box1 or not box2:
        return 1000.0

    cx1, cy1 = calculate_box_center(box1)
    cx2, cy2 = calculate_box_center(box2)
    return math.hypot(cx1 - cx2, cy1 - cy2)


def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculates Intersection over Union (IoU) ratio between two 0-1000 scale bounding boxes.
    """
    if not box1 or not box2:
        return 0.0

    x_left = max(box1['x_min'], box2['x_min'])
    y_top = max(box1['y_min'], box2['y_min'])
    x_right = min(box1['x_max'], box2['x_max'])
    y_bottom = min(box1['y_max'], box2['y_max'])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])

    union_area = box1_area + box2_area - intersection_area
    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


def calculate_containment(box1: Dict[str, float], box2: Dict[str, float]) -> float:
    """
    Calculates containment ratio (intersection area over smaller box area)
    between two 0-1000 scale bounding boxes.
    """
    if not box1 or not box2:
        return 0.0

    x_left = max(box1['x_min'], box2['x_min'])
    y_top = max(box1['y_min'], box2['y_min'])
    x_right = min(box1['x_max'], box2['x_max'])
    y_bottom = min(box1['y_max'], box2['y_max'])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1['x_max'] - box1['x_min']) * (box1['y_max'] - box1['y_min'])
    box2_area = (box2['x_max'] - box2['x_min']) * (box2['y_max'] - box2['y_min'])
    min_area = min(box1_area, box2_area)

    if min_area <= 0:
        return 0.0

    return intersection_area / min_area


def compute_image_transformation_metadata(
    orig_w: int,
    orig_h: int,
    analysis_w: int,
    analysis_h: int
) -> Dict[str, Any]:
    """
    Constructs comprehensive transformation metadata between original image
    and preprocessed VLM analysis image.
    """
    scale_x = float(analysis_w) / float(orig_w) if orig_w > 0 else 1.0
    scale_y = float(analysis_h) / float(orig_h) if orig_h > 0 else 1.0

    return {
        "original_width": orig_w,
        "original_height": orig_h,
        "analysis_width": analysis_w,
        "analysis_height": analysis_h,
        "scale_x": round(scale_x, 6),
        "scale_y": round(scale_y, 6),
        "offset_x": 0,
        "offset_y": 0,
        "rotation": 0,
        "crop": None
    }

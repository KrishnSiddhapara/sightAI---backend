"""
Centralized Bounding Box & Image Coordinate Transformation Utility for SightAI Backend.

Handles:
- Box normalization (0.0-1.0 or 0-1000 scale) with whole-box scale detection.
- Box inversion repair & clamping.
- Image transformation metadata generation.
"""

from typing import Optional, Dict, Any, Union, List, Tuple
import logging

logger = logging.getLogger(__name__)

def parse_and_normalize_bbox(raw_box: Any) -> Optional[Dict[str, float]]:
    """
    Parses raw bounding box data returned by LLM/VLM into a standardized dict:
    {'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float} in 0-1000 scale.
    
    Handles:
    - 4-element lists/tuples [ymin, xmin, ymax, xmax] (Google 2D bounding box format) or [xmin, ymin, xmax, ymax]
    - Dictionaries with x_min/xmin/xMin, y_min/ymin/yMin, x_max/xmax/xMax, y_max/ymax/yMax
    - Whole-box scale detection (0.0-1.0 vs 0-1000) to prevent corrupting single coordinates.
    """
    if raw_box is None:
        return None

    x1, y1, x2, y2 = None, None, None, None

    # Case A: List or Tuple [4 elements]
    if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
        try:
            vals = [float(v) for v in raw_box]
            # Heuristic format detection: try both orderings and pick the valid one.
            # Google box_2d format: [ymin, xmin, ymax, xmax]
            # Standard format:      [xmin, ymin, xmax, ymax]
            
            # Try standard [xmin, ymin, xmax, ymax] first
            sx1, sy1, sx2, sy2 = vals[0], vals[1], vals[2], vals[3]
            standard_valid = (sx2 > sx1 and sy2 > sy1)
            
            # Try Google [ymin, xmin, ymax, xmax]
            gy1, gx1, gy2, gx2 = vals[0], vals[1], vals[2], vals[3]
            google_valid = (gx2 > gx1 and gy2 > gy1)
            
            if standard_valid and not google_valid:
                x1, y1, x2, y2 = sx1, sy1, sx2, sy2
            elif google_valid and not standard_valid:
                x1, y1, x2, y2 = gx1, gy1, gx2, gy2
            elif standard_valid and google_valid:
                # Both valid — prefer standard [x,y,x,y] since our prompt explicitly asks for x_min,y_min,x_max,y_max
                x1, y1, x2, y2 = sx1, sy1, sx2, sy2
            else:
                # Neither valid as-is, use standard order and let min/max repair handle it
                x1, y1, x2, y2 = sx1, sy1, sx2, sy2
        except (ValueError, TypeError):
            return None

    # Case B: Dictionary format
    elif isinstance(raw_box, dict):
        # Look for keys
        raw_x_min = raw_box.get('x_min') if raw_box.get('x_min') is not None else raw_box.get('xmin', raw_box.get('xMin'))
        raw_y_min = raw_box.get('y_min') if raw_box.get('y_min') is not None else raw_box.get('ymin', raw_box.get('yMin'))
        raw_x_max = raw_box.get('x_max') if raw_box.get('x_max') is not None else raw_box.get('xmax', raw_box.get('xMax'))
        raw_y_max = raw_box.get('y_max') if raw_box.get('y_max') is not None else raw_box.get('ymax', raw_box.get('yMax'))

        if all(v is not None for v in [raw_x_min, raw_y_min, raw_x_max, raw_y_max]):
            try:
                x1, y1, x2, y2 = float(raw_x_min), float(raw_y_min), float(raw_x_max), float(raw_y_max)
            except (ValueError, TypeError):
                return None

    if x1 is None or y1 is None or x2 is None or y2 is None:
        return None

    # Check for NaN / Infinity
    if any(val != val or abs(val) == float('inf') for val in [x1, y1, x2, y2]):
        return None

    # WHOLE-BOX Scale Detection:
    # Check if ALL 4 non-zero values are <= 1.0 (indicating 0.0-1.0 normalized scale)
    is_zero_one_scale = all(0.0 <= v <= 1.0 for v in [x1, y1, x2, y2])

    if is_zero_one_scale:
        x1 *= 1000.0
        y1 *= 1000.0
        x2 *= 1000.0
        y2 *= 1000.0

    # Ensure min <= max
    xmin = min(x1, x2)
    xmax = max(x1, x2)
    ymin = min(y1, y2)
    ymax = max(y1, y2)

    # Clamp to [0, 1000]
    xmin = max(0.0, min(1000.0, xmin))
    ymin = max(0.0, min(1000.0, ymin))
    xmax = max(0.0, min(1000.0, xmax))
    ymax = max(0.0, min(1000.0, ymax))

    return {
        'x_min': round(xmin, 2),
        'y_min': round(ymin, 2),
        'x_max': round(xmax, 2),
        'y_max': round(ymax, 2)
    }

def sanitize_box(x_min: float, y_min: float, x_max: float, y_max: float, min_size_px_in_1000: float = 2.0) -> Optional[Dict[str, float]]:
    """
    Validates and sanitizes 0-1000 scale bounding box coordinates.
    Repairs inversions and discards zero/tiny area boxes (< min_size_px_in_1000 scale).
    """
    try:
        x1 = max(0.0, min(1000.0, float(x_min)))
        y1 = max(0.0, min(1000.0, float(y_min)))
        x2 = max(0.0, min(1000.0, float(x_max)))
        y2 = max(0.0, min(1000.0, float(y_max)))

        xmin = min(x1, x2)
        xmax = max(x1, x2)
        ymin = min(y1, y2)
        ymax = max(y1, y2)

        # Minimum dimensions check in 1000 scale (default 2.0 = 0.2% of image dimension)
        if (xmax - xmin) >= min_size_px_in_1000 and (ymax - ymin) >= min_size_px_in_1000:
            return {
                'x_min': round(xmin, 2),
                'y_min': round(ymin, 2),
                'x_max': round(xmax, 2),
                'y_max': round(ymax, 2)
            }
        else:
            return None
    except (ValueError, TypeError):
        return None

def compute_image_transformation_metadata(
    orig_w: int,
    orig_h: int,
    analysis_w: int,
    analysis_h: int
) -> Dict[str, Any]:
    """
    Constructs comprehensive transformation metadata between original image and preprocessed VLM analysis image.
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
    Calculates containment ratio (intersection area over smaller box area) between two 0-1000 scale bounding boxes.
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


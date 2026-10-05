import io
import datetime
import logging
from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
from PIL import Image

logger = logging.getLogger(__name__)

def create_initial_version(original_image: Image.Image) -> Dict[str, Any]:
    """
    Creates Version 0 (Original Image) version record.
    Ensures image object is copied/isolated to prevent unintended mutations.
    Generates backend ISO 8601 timezone-aware server timestamp.
    """
    if not isinstance(original_image, Image.Image):
        raise ValueError("Invalid original image input.")

    # Create an independent copy in RGB mode
    img_copy = original_image.copy()
    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")

    now_dt = datetime.datetime.now().astimezone()
    iso_timestamp = now_dt.isoformat()
    formatted_time = now_dt.strftime("%d %b %Y • %I:%M %p")

    return {
        "version_id": "v0",
        "version_number": 0,
        "image": img_copy,
        "edit_prompt": "Original Image",
        "source_version_number": 0,
        "parent_version_id": "v0",
        "created_at": iso_timestamp,
        "updated_at": iso_timestamp,
        "formatted_time": formatted_time
    }

def add_new_version(
    history: List[Dict[str, Any]],
    new_image: Image.Image,
    prompt: str,
    source_version_num: int,
) -> Dict[str, Any]:
    """
    Appends a new immutable version record to version history with backend server timestamp.
    
    Args:
        history: Current list of version records.
        new_image: PIL Image returned from editing service.
        prompt: User edit instruction used.
        source_version_num: Version number of the base image used for this edit.
        
    Returns:
        The newly created version record dictionary.
    """
    if not history:
        raise ValueError("Cannot add version: history is uninitialized.")

    if not isinstance(new_image, Image.Image):
        raise ValueError("Invalid generated image input.")

    # Determine next version number
    max_num = max(v.get("version_number", 0) for v in history)
    next_num = max_num + 1

    # Isolate image copy in RGB mode
    img_copy = new_image.copy()
    if img_copy.mode != "RGB":
        img_copy = img_copy.convert("RGB")

    now_dt = datetime.datetime.now().astimezone()
    iso_timestamp = now_dt.isoformat()
    formatted_time = now_dt.strftime("%d %b %Y • %I:%M %p")

    new_record = {
        "version_id": f"v{next_num}",
        "version_number": next_num,
        "image": img_copy,
        "edit_prompt": prompt.strip(),
        "source_version_number": source_version_num,
        "parent_version_id": f"v{source_version_num}",
        "created_at": iso_timestamp,
        "updated_at": iso_timestamp,
        "formatted_time": formatted_time
    }

    history.append(new_record)
    logger.info(f"Created version v{next_num} based on v{source_version_num} at {iso_timestamp} with prompt: '{prompt.strip()}'")
    return new_record

def get_version_by_number(
    history: List[Dict[str, Any]],
    version_num: int
) -> Optional[Dict[str, Any]]:
    """
    Finds and returns version record matching version_num, or None if not found.
    """
    for v in history:
        if v.get("version_number") == version_num:
            return v
    return None

def get_latest_version(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Returns the latest created version record in history.
    """
    if not history:
        return None
    return max(history, key=lambda v: v.get("version_number", 0))

def export_image_bytes(image: Image.Image, format_type: str = "JPEG") -> bytes:
    """
    Converts a PIL Image object to raw bytes (JPEG, PNG, or PDF) for file export and download responses.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image for byte export.")

    img_rgb = image.convert("RGB") if image.mode != "RGB" else image
    buf = io.BytesIO()

    fmt_upper = format_type.upper()
    if fmt_upper in ["JPG", "JPEG"]:
        img_rgb.save(buf, format="JPEG", quality=95)
    elif fmt_upper == "PDF":
        img_rgb.save(buf, format="PDF", resolution=100.0)
    elif fmt_upper == "PNG":
        img_rgb.save(buf, format="PNG")
    else:
        img_rgb.save(buf, format="JPEG", quality=95)

    return buf.getvalue()

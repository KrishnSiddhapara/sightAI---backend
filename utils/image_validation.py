import hashlib
import io
from typing import Tuple, Optional
from PIL import Image, ImageFile, ImageOps

# Allow loading truncated images safely where possible, but verify integrity
ImageFile.LOAD_TRUNCATED_IMAGES = False

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

def compute_image_hash(file_bytes: bytes) -> str:
    """
    Computes SHA-256 hash of image bytes for robust identity tracking in session state.
    """
    return hashlib.sha256(file_bytes).hexdigest()

def validate_image_file(
    uploaded_file,
    max_size_mb: float = 10.0
) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validates uploaded image file format, size limit, and PIL image integrity.
    Applies EXIF auto-rotation (ImageOps.exif_transpose) and standard RGB conversion
    to ensure full visual accuracy.
    
    Returns:
        (is_valid, error_message, pil_image)
    """
    if uploaded_file is None:
        return False, "No file was uploaded.", None

    # Check file size limit
    file_bytes = uploaded_file.getvalue()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_size_mb:
        return (
            False,
            f"File size exceeds limit of {max_size_mb:.1f} MB (uploaded file is {size_mb:.1f} MB). Please choose a smaller image.",
            None,
        )

    # Check file extension
    file_name = uploaded_file.name.lower()
    has_valid_ext = any(file_name.endswith(ext) for ext in ALLOWED_EXTENSIONS)
    if not has_valid_ext:
        return (
            False,
            "Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP image.",
            None,
        )

    # Validate image integrity with PIL
    try:
        stream = io.BytesIO(file_bytes)
        img = Image.open(stream)
        
        # Verify image structure
        img.verify()
        
        # Re-open stream for actual usage because verify() clears image state
        stream.seek(0)
        usable_img = Image.open(stream)
        usable_img.load()  # Force loading pixel data into memory

        # Auto-transpose EXIF orientation metadata so rotated photos display correctly
        try:
            usable_img = ImageOps.exif_transpose(usable_img)
        except Exception:
            pass  # If no EXIF data, continue safely
        
        # Ensure conversion to standard RGB mode
        if usable_img.mode != "RGB":
            usable_img = usable_img.convert("RGB")
            
        return True, None, usable_img
    except Exception as e:
        return (
            False,
            "The uploaded file appears to be corrupted or is not a valid image.",
            None,
        )

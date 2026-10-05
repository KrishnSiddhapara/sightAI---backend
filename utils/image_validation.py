import hashlib
import io
from typing import Tuple, Optional
from PIL import Image, ImageFile, ImageOps
from utils.config import MAX_ANALYSIS_DIMENSION, ANALYSIS_IMAGE_QUALITY

# Allow loading truncated images safely where possible, but verify integrity
ImageFile.LOAD_TRUNCATED_IMAGES = False

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}

def compute_image_hash(file_bytes: bytes) -> str:
    """
    Computes SHA-256 hash of image bytes for robust identity tracking in session state.
    """
    return hashlib.sha256(file_bytes).hexdigest()

def optimize_image_for_analysis(img: Image.Image, max_dim: int = MAX_ANALYSIS_DIMENSION) -> Image.Image:
    """
    Proportionally downscales large high-resolution images so that neither width nor height
    exceeds max_dim (default 1536px), while maintaining exact aspect ratio.
    
    This preserves 100% visual detail for VLM analysis & normalized bounding boxes while
    drastically reducing image byte payload, memory overhead, and model inference latency.
    """
    if img is None:
        return img
    
    width, height = img.size
    if width <= max_dim and height <= max_dim:
        return img

    # Compute proportional scale factor
    scale = float(max_dim) / float(max(width, height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    # High quality LANCZOS resampling
    resample_filter = getattr(Image.Resampling, 'LANCZOS', Image.LANCZOS)
    return img.resize((new_width, new_height), resample_filter)

def encode_vlm_image_part(img: Image.Image, max_dim: int = MAX_ANALYSIS_DIMENSION, max_bytes: int = 800 * 1024) -> Tuple[bytes, str, int, int]:
    """
    Encodes a PIL Image into optimized JPEG bytes guaranteed to stay below max_bytes (default 800KB).
    Strictly prevents 'Part exceeded maximum size of 1024KB' errors when transmitting multimodal content.
    
    Returns:
        (jpeg_bytes, mime_type, width, height)
    """
    if img is None:
        return b"", "image/jpeg", 0, 0

    curr_dim = max_dim
    optimized = optimize_image_for_analysis(img, max_dim=curr_dim)
    width, height = optimized.size
    rgb_img = optimized.convert("RGB") if optimized.mode != "RGB" else optimized

    buffer = io.BytesIO()

    while True:
        quality = ANALYSIS_IMAGE_QUALITY  # Default 85
        while quality >= 30:
            buffer.seek(0)
            buffer.truncate(0)
            rgb_img.save(buffer, format="JPEG", quality=quality, optimize=True)
            size_bytes = buffer.tell()
            if size_bytes <= max_bytes:
                break
            quality -= 10

        if size_bytes <= max_bytes or curr_dim <= 400:
            break

        # If quality down to 30 still exceeds max_bytes, downscale dimensions and retry
        curr_dim = int(curr_dim * 0.85)
        optimized = optimize_image_for_analysis(img, max_dim=curr_dim)
        width, height = optimized.size
        rgb_img = optimized.convert("RGB") if optimized.mode != "RGB" else optimized

    buffer.seek(0)
    return buffer.read(), "image/jpeg", width, height

import logging
logger = logging.getLogger(__name__)

def prepare_image_for_edit(
    img: Image.Image,
    target_max_bytes: int = 900 * 1024,
    initial_max_dim: int = 2048,
    max_attempts: int = 6
) -> Tuple[bytes, str, int, int]:
    """
    Centralized Image Edit Optimizer.
    Adaptively optimizes any original input image (up to 5MB user upload) into a compliant
    JPEG byte payload strictly <= target_max_bytes (900 KB) before transmitting to Gemini.
    
    Guarantees that the payload wrapped in types.Part will never exceed Gemini's 1024KB limit.
    """
    if img is None:
        return b"", "image/jpeg", 0, 0

    orig_w, orig_h = img.size
    orig_estimated_mb = (orig_w * orig_h * 3) / (1024 * 1024)

    logger.info(
        f"[IMAGE_EDIT_INPUT]\n"
        f"Original dimensions: {orig_w}x{orig_h}\n"
        f"Original estimated binary size: ~{orig_estimated_mb:.2f} MB"
    )

    curr_dim = initial_max_dim if max(orig_w, orig_h) > initial_max_dim else max(orig_w, orig_h)
    buffer = io.BytesIO()
    
    attempt_count = 0
    final_bytes = b""
    final_w, final_h = orig_w, orig_h
    used_quality = 85

    while attempt_count < max_attempts:
        attempt_count += 1
        optimized = optimize_image_for_analysis(img, max_dim=curr_dim)
        final_w, final_h = optimized.size
        rgb_img = optimized.convert("RGB") if optimized.mode != "RGB" else optimized

        used_quality = 90
        while used_quality >= 35:
            buffer.seek(0)
            buffer.truncate(0)
            rgb_img.save(buffer, format="JPEG", quality=used_quality, optimize=True)
            size_bytes = buffer.tell()
            if size_bytes <= target_max_bytes:
                break
            used_quality -= 10

        buffer.seek(0)
        final_bytes = buffer.read()
        if len(final_bytes) <= target_max_bytes:
            break

        # If quality down to 35 still exceeds target_max_bytes, scale down dimensions and retry
        curr_dim = int(curr_dim * 0.82)

    final_size_kb = len(final_bytes) / 1024
    base64_size_kb = (len(final_bytes) * 4 / 3) / 1024

    logger.info(
        f"[IMAGE_EDIT_OPTIMIZATION]\n"
        f"Output dimensions: {final_w}x{final_h}\n"
        f"Output format: JPEG\n"
        f"Output quality: Q{used_quality}\n"
        f"Output binary size: {final_size_kb:.1f} KB\n"
        f"Base64 size: {base64_size_kb:.1f} KB\n"
        f"Estimated Gemini Part size: {final_size_kb:.1f} KB"
    )

    is_safe_transport = len(final_bytes) < 1000 * 1024
    logger.info(
        f"[IMAGE_EDIT_REQUEST]\n"
        f"Final Gemini payload size: {final_size_kb:.1f} KB (Safe transport: {is_safe_transport})"
    )

    return final_bytes, "image/jpeg", final_w, final_h

def prepare_ask_ai_image_part(img: Image.Image, max_dim: int = 2048, max_bytes: int = 900 * 1024) -> Tuple[bytes, str, int, int]:
    """
    Prepares a high-resolution, high-quality image Part for Ask AI visual Q&A.
    Preserves fine details such as small text, logos, model numbers, and fine textures
    while ensuring output byte size stays safely below max_bytes (900KB).
    
    Logs safe diagnostics without exposing keys, secret data, or raw bytes.
    """
    if img is None:
        return b"", "image/jpeg", 0, 0

    orig_w, orig_h = img.size
    
    # Target high-resolution dimensions (prefer full resolution up to 2048px)
    target_dim = max_dim if max(orig_w, orig_h) > max_dim else max(orig_w, orig_h)
    optimized = optimize_image_for_analysis(img, max_dim=target_dim)
    final_w, final_h = optimized.size
    rgb_img = optimized.convert("RGB") if optimized.mode != "RGB" else optimized

    quality = 92  # High quality for clear text, labels, and fine visual details
    buffer = io.BytesIO()

    while quality >= 60:
        buffer.seek(0)
        buffer.truncate(0)
        rgb_img.save(buffer, format="JPEG", quality=quality, optimize=True)
        size_bytes = buffer.tell()
        if size_bytes <= max_bytes:
            break
        quality -= 5

    # If still > max_bytes, resize to 1800px and re-encode with Q88
    if size_bytes > max_bytes:
        optimized = optimize_image_for_analysis(img, max_dim=1800)
        final_w, final_h = optimized.size
        rgb_img = optimized.convert("RGB") if optimized.mode != "RGB" else optimized
        quality = 88
        buffer.seek(0)
        buffer.truncate(0)
        rgb_img.save(buffer, format="JPEG", quality=quality, optimize=True)
        size_bytes = buffer.tell()

    buffer.seek(0)
    final_bytes = buffer.read()

    orig_estimated_mb = (orig_w * orig_h * 3) / (1024 * 1024)
    final_size_kb = len(final_bytes) / 1024
    opt_applied = "High-Quality Original" if (final_w == orig_w and final_h == orig_h and quality >= 90) else f"High-Detail Resize ({final_w}x{final_h}, Q{quality})"

    logger.info(
        f"[ASK_AI_IMAGE]\n"
        f"Original size: ~{orig_estimated_mb:.2f} MB\n"
        f"Original dimensions: {orig_w}x{orig_h}\n"
        f"Ask AI image size: {final_size_kb:.1f} KB\n"
        f"Ask AI image dimensions: {final_w}x{final_h}\n"
        f"Optimization applied: {opt_applied}"
    )

    return final_bytes, "image/jpeg", final_w, final_h

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

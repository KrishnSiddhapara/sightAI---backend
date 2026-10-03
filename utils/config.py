import os

# SightAI Backend Configuration Settings

# Maximum upload file size limit (in MB)
MAX_UPLOAD_SIZE_MB: float = float(os.getenv("MAX_UPLOAD_SIZE_MB", "10.0"))

# Maximum pixel dimension (width or height) for VLM image analysis optimization.
# Downscaling high-res images (e.g. 8000x6000 -> 1536x1024) drastically reduces
# network payload size and VLM tokenization latency while preserving 100% visual detail.
MAX_ANALYSIS_DIMENSION: int = int(os.getenv("MAX_ANALYSIS_DIMENSION", "1536"))

# Image quality for JPEG conversion during analysis optimization
ANALYSIS_IMAGE_QUALITY: int = int(os.getenv("ANALYSIS_IMAGE_QUALITY", "85"))

# Maximum output tokens for structured Gemini vision responses
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

# Request timeout limit in seconds
ANALYSIS_TIMEOUT: float = float(os.getenv("ANALYSIS_TIMEOUT", "120.0"))

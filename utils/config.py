import os

# SightAI Backend Configuration Settings

# Maximum upload file size limit (in MB)
MAX_UPLOAD_SIZE_MB: float = float(os.getenv("MAX_UPLOAD_SIZE_MB", "10.0"))

# Maximum pixel dimension (width or height) for VLM image analysis optimization.
MAX_ANALYSIS_DIMENSION: int = int(os.getenv("MAX_ANALYSIS_DIMENSION", "1536"))

# Image quality for JPEG conversion during analysis optimization
ANALYSIS_IMAGE_QUALITY: int = int(os.getenv("ANALYSIS_IMAGE_QUALITY", "85"))

# Maximum output tokens for structured Gemini vision responses
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "4096"))

# Request timeout limit in seconds
ANALYSIS_TIMEOUT: float = float(os.getenv("ANALYSIS_TIMEOUT", "120.0"))

# Primary VLM model configurable via .env (default: gemini-2.5-flash)
PRIMARY_VLM_MODEL: str = os.getenv("VLM_MODEL", "gemini-2.5-flash").strip()

# Fallback VLM model if primary model experiences 503 capacity issues
FALLBACK_VLM_MODEL: str = os.getenv("FALLBACK_VLM_MODEL", "gemini-2.0-flash").strip()

# Debug logging flag
DEBUG_ANALYSIS: bool = os.getenv("DEBUG_ANALYSIS", "true").lower() in ["true", "1", "yes"]

# Allowed Frontend CORS Origins
FRONTEND_ORIGINS_ENV: str = os.getenv("FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
FRONTEND_ORIGINS = [origin.strip() for origin in FRONTEND_ORIGINS_ENV.split(",") if origin.strip()]
if "*" not in FRONTEND_ORIGINS:
    FRONTEND_ORIGINS.append("*")


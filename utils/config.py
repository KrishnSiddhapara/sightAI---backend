import os

# SightAI Backend Configuration Settings

# Maximum upload file size limit for general image analysis (in MB)
MAX_UPLOAD_SIZE_MB: float = float(os.getenv("MAX_UPLOAD_SIZE_MB", "10.0"))

# Maximum upload file size limit for AI Image Editing (User Upload Limit: 5.0 MB)
AI_EDITOR_MAX_IMAGE_SIZE_MB: float = float(os.getenv("AI_EDITOR_MAX_IMAGE_SIZE_MB", "5.0"))
AI_EDITOR_MAX_IMAGE_SIZE_BYTES: int = int(AI_EDITOR_MAX_IMAGE_SIZE_MB * 1024 * 1024)

# Safe Gemini Transport Part Limit (in Bytes: 900 KB target to strictly stay under Gemini 1024 KB inline Part limit)
GEMINI_PART_SAFE_LIMIT_BYTES: int = 900 * 1024

# Maximum pixel dimension (width or height) for VLM image analysis optimization.
MAX_ANALYSIS_DIMENSION: int = int(os.getenv("MAX_ANALYSIS_DIMENSION", "1536"))

# Image quality for JPEG conversion during analysis optimization
ANALYSIS_IMAGE_QUALITY: int = int(os.getenv("ANALYSIS_IMAGE_QUALITY", "85"))

# Maximum output tokens for structured Gemini vision responses (8192 for full JSON budget)
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))

# Request timeout limit in seconds (kept under Render's 100s proxy timeout to prevent 502 errors)
ANALYSIS_TIMEOUT: float = float(os.getenv("ANALYSIS_TIMEOUT", "45.0"))

# Primary VLM model (strictly gemini-2.5-flash)
PRIMARY_VLM_MODEL: str = os.getenv("VLM_MODEL", "gemini-2.5-flash").strip()

# Fallback model set strictly to gemini-2.5-flash
FALLBACK_VLM_MODEL: str = "gemini-2.5-flash"

# Debug logging flag
DEBUG_ANALYSIS: bool = os.getenv("DEBUG_ANALYSIS", "true").lower() in ["true", "1", "yes"]

# Allowed Frontend CORS Origins
DEFAULT_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "https://sightai-frontend.vercel.app",
]

FRONTEND_ORIGINS_ENV: str = os.getenv("FRONTEND_ORIGINS", "").strip()
if FRONTEND_ORIGINS_ENV:
    configured_origins = [origin.strip().rstrip("/") for origin in FRONTEND_ORIGINS_ENV.split(",") if origin.strip()]
    FRONTEND_ORIGINS = list(dict.fromkeys(configured_origins + DEFAULT_ORIGINS))
else:
    FRONTEND_ORIGINS = DEFAULT_ORIGINS


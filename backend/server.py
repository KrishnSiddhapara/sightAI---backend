import os
import io
import json
import base64
import logging
import time
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
from PIL import Image
from google.genai import types

# Import existing core Python AI service modules & utilities
from utils.image_validation import validate_image_file, compute_image_hash, optimize_image_for_analysis, encode_vlm_image_part
from utils.config import MAX_ANALYSIS_DIMENSION, ANALYSIS_TIMEOUT, FRONTEND_ORIGINS
from services.safety import check_image_safety
from services.vision import analyze_image_grounded
from services.user_query import answer_image_query
from services.image_editor import edit_image, check_edit_instruction_ambiguity, validate_edit_instruction
from services.version_manager import export_image_bytes
from services.schemas import GroundedAnalysisResult, AgentResearchRequest, AgentResearchResponse
from services.research_agent import execute_agent_research

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend_api")

app = FastAPI(
    title="AI Image Object Identifier API",
    description="REST API backend for AI Image Object Identifier, grounded VLM analysis, safety gate, multi-version editing, and visual Q&A.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DummyUploadedFile:
    """Wrapper to make raw bytes compatible with validate_image_file utility."""
    def __init__(self, filename: str, content_bytes: bytes):
        self.name = filename
        self._bytes = content_bytes
        
    def getvalue(self) -> bytes:
        return self._bytes

def get_api_key() -> str:
    key = os.getenv("VLM_API_KEY", "").strip()
    if not key or key == "your_api_key_here":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Gemini API Key (VLM_API_KEY) is not properly configured on the backend server."
        )
    return key

def parse_and_validate_file(filename: str, content_bytes: bytes) -> Image.Image:
    """Uses existing validate_image_file utility to validate and extract PIL Image."""
    dummy_file = DummyUploadedFile(filename, content_bytes)
    is_valid, err_msg, pil_img = validate_image_file(dummy_file, max_size_mb=10.0)
    if not is_valid or pil_img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=err_msg or "Invalid image file."
        )
    return pil_img

def pil_to_base64(img: Image.Image, format_type: str = "JPEG") -> str:
    """Converts a PIL Image to a Base64 data URI string."""
    buffer = io.BytesIO()
    fmt_upper = "JPEG" if format_type.upper() in ["JPG", "JPEG"] else format_type.upper()
    img_rgb = img.convert("RGB") if img.mode != "RGB" else img
    img_rgb.save(buffer, format=fmt_upper, quality=95)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/jpeg" if fmt_upper == "JPEG" else f"image/{fmt_upper.lower()}"
    return f"data:{mime};base64,{b64_str}"

def base64_to_pil(b64_str: str) -> Image.Image:
    """Decodes a Base64 data URI string back into a PIL Image."""
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(image_bytes))
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img

@app.get("/api/health")
def health_check():
    key = os.getenv("VLM_API_KEY", "").strip()
    is_key_configured = bool(key and key != "your_api_key_here")
    return {
        "status": "healthy",
        "api_key_configured": is_key_configured,
        "message": "AI Image Object Identifier backend API is running."
    }

@app.post("/api/safety-check")
async def safety_check_endpoint(file: UploadFile = File(...)):
    content_bytes = await file.read()
    pil_img = parse_and_validate_file(file.filename or "uploaded_image.jpg", content_bytes)
    
    api_key = get_api_key()
    safety_result = check_image_safety(pil_img, api_key)
    
    return {
        "success": True,
        "image_hash": compute_image_hash(content_bytes),
        "safety": safety_result
    }

@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    import uuid
    req_id = f"req_{uuid.uuid4().hex[:8]}"
    t_start = time.perf_counter()
    safety_result = None

    try:
        t0_read = time.perf_counter()
        content_bytes = await file.read()
        filename = file.filename or "uploaded_image.jpg"
        t_upload = time.perf_counter() - t0_read

        size_mb = len(content_bytes) / (1024 * 1024)
        logger.info(f"[{req_id}] [UPLOAD] Received '{filename}' ({size_mb:.2f} MB)")

        # Step 1: Decode & validate original image
        t0_decode = time.perf_counter()
        pil_img = parse_and_validate_file(filename, content_bytes)
        t_decode = time.perf_counter() - t0_decode
        orig_w, orig_h = pil_img.size
        logger.info(f"[{req_id}] [IMAGE] Original: {size_mb:.2f} MB ({orig_w}x{orig_h})")

        api_key = get_api_key()

        # Step 2: Preprocess & encode VLM image part ONCE for both safety and vision calls
        t0_opt = time.perf_counter()
        vlm_bytes, mime_type, opt_w, opt_h = encode_vlm_image_part(pil_img, MAX_ANALYSIS_DIMENSION)
        image_part = types.Part.from_bytes(data=vlm_bytes, mime_type=mime_type)
        t_opt = time.perf_counter() - t0_opt

        logger.info(f"[{req_id}] [ANALYSIS IMAGE] Dimensions: {opt_w}x{opt_h}, JPEG size: {len(vlm_bytes)/1024:.1f} KB, MIME: {mime_type} (prep time: {t_opt:.3f}s)")

        # Step 3: Safety Gate Pre-screening (uses pre-encoded image_part)
        t0_safety = time.perf_counter()
        safety_result = check_image_safety(image_part, api_key)
        t_safety = time.perf_counter() - t0_safety
        logger.info(f"[{req_id}] [SAFETY] Screening completed (is_safe={safety_result.get('is_safe')}) in {t_safety:.3f}s")

        if not safety_result.get("is_safe", False):
            logger.warning(f"[{req_id}] [SAFETY_FLAGGED] Image rejected: {safety_result.get('category')}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "request_id": req_id,
                    "error_code": safety_result.get("error_code") or "SAFETY_FLAGGED",
                    "error": safety_result.get("error") or "Image rejected by safety screening gate.",
                    "safety": safety_result
                }
            )

        # Step 4: Grounded VLM Analysis (uses pre-encoded image_part)
        t0_vlm = time.perf_counter()
        logger.info(f"[{req_id}] [VLM_REQUEST] Sending multimodal request to Gemini VLM API")
        grounded_result: GroundedAnalysisResult = analyze_image_grounded(image_part, api_key)
        t_vlm = time.perf_counter() - t0_vlm
        t_total = time.perf_counter() - t_start

        logger.info(
            f"[{req_id}] [PERF_BREAKDOWN] "
            f"UPLOAD: {t_upload:.3f}s | "
            f"VALIDATION: {t_decode:.3f}s | "
            f"PREPROCESSING: {t_opt:.3f}s | "
            f"SAFETY: {t_safety:.3f}s | "
            f"VISION: {t_vlm:.3f}s | "
            f"TOTAL: {t_total:.3f}s"
        )
        logger.info(f"[{req_id}] [FINAL_RESPONSE] Result ready (Objects={len(grounded_result.objects)}, Scene='{grounded_result.scene.category}')")

        return {
            "success": True,
            "request_id": req_id,
            "image_hash": compute_image_hash(content_bytes),
            "safety": safety_result,
            "data": grounded_result.model_dump(),
            "performance": {
                "total_seconds": round(t_total, 3),
                "vlm_seconds": round(t_vlm, 3),
                "safety_seconds": round(t_safety, 3),
                "preprocessing_seconds": round(t_opt, 3),
                "original_dimensions": [orig_w, orig_h],
                "analysis_dimensions": [opt_w, opt_h],
                "image_bytes_kb": round(len(vlm_bytes) / 1024, 1)
            }
        }
    except HTTPException as http_ex:
        t_total = time.perf_counter() - t_start
        logger.warning(f"[{req_id}] [HTTP_ERROR] {http_ex.status_code}: {http_ex.detail} after {t_total:.3f}s")
        return JSONResponse(
            status_code=http_ex.status_code,
            content={
                "success": False,
                "request_id": req_id,
                "error_code": "HTTP_ERROR",
                "error": http_ex.detail,
                "safety": safety_result
            }
        )
    except (TimeoutError, ConnectionError, ValueError, Exception) as e:
        t_total = time.perf_counter() - t_start
        err_msg = str(e)
        error_code = "INTERNAL_ERROR"

        if "GEMINI_TIMEOUT" in err_msg or isinstance(e, TimeoutError):
            error_code = "GEMINI_TIMEOUT"
        elif "GEMINI_CAPACITY" in err_msg:
            error_code = "GEMINI_CAPACITY"
        elif "NETWORK_ERROR" in err_msg or isinstance(e, ConnectionError):
            error_code = "NETWORK_ERROR"
        elif "API_KEY_ERROR" in err_msg:
            error_code = "API_KEY_ERROR"
        elif "SCHEMA_VALIDATION_ERROR" in err_msg:
            error_code = "SCHEMA_VALIDATION_ERROR"
        elif "INVALID_IMAGE" in err_msg:
            error_code = "INVALID_IMAGE"

        logger.error(f"[{req_id}] [VLM_ERROR] ({error_code}) after {t_total:.3f}s: {err_msg}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "success": False,
                "request_id": req_id,
                "error_code": error_code,
                "error": f"Image analysis encountered an issue ({err_msg}). Please retry.",
                "safety": safety_result
            }
        )

@app.post("/api/edit")
async def edit_image_endpoint(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    instruction: str = Form(...),
    vision_context_json: Optional[str] = Form(None)
):
    api_key = get_api_key()
    
    if file is not None:
        content_bytes = await file.read()
        pil_img = parse_and_validate_file(file.filename or "input.jpg", content_bytes)
    elif image_base64 is not None:
        try:
            pil_img = base64_to_pil(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either file upload or image_base64 must be provided.")

    val_err = validate_edit_instruction(instruction)
    if val_err:
        raise HTTPException(status_code=400, detail=val_err)

    vision_ctx = None
    if vision_context_json:
        try:
            parsed_ctx = json.loads(vision_context_json)
            vision_ctx = GroundedAnalysisResult.model_validate(parsed_ctx)
        except Exception as e:
            logger.warning(f"Could not parse vision context JSON: {e}")

    # Check target ambiguity helper
    is_ambig, ambig_msg = check_edit_instruction_ambiguity(instruction, vision_ctx)

    try:
        edited_pil = edit_image(
            image=pil_img,
            instruction=instruction,
            api_key=api_key,
            vision_context=vision_ctx
        )

        # Safety Gate screening on generated edit
        edit_safety_result = check_image_safety(edited_pil, api_key)
        if not edit_safety_result.get("is_safe", False):
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "is_safe": False,
                    "error": edit_safety_result.get("error") or "Generated edit contained unsafe content and was discarded.",
                    "safety": edit_safety_result,
                    "ambiguity_warning": ambig_msg if is_ambig else None
                }
            )

        edited_b64 = pil_to_base64(edited_pil, "JPEG")
        return {
            "success": True,
            "is_safe": True,
            "image_base64": edited_b64,
            "prompt": instruction.strip(),
            "safety": edit_safety_result,
            "ambiguity_warning": ambig_msg if is_ambig else None
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error during image edit: {e}")
        raise HTTPException(status_code=500, detail=f"Image edit failed: {str(e)}")

@app.post("/api/ask")
async def ask_question_endpoint(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    question: str = Form(...)
):
    api_key = get_api_key()

    if file is not None:
        content_bytes = await file.read()
        pil_img = parse_and_validate_file(file.filename or "input.jpg", content_bytes)
    elif image_base64 is not None:
        try:
            pil_img = base64_to_pil(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="Either file upload or image_base64 must be provided.")

    try:
        opt_img = optimize_image_for_analysis(pil_img, MAX_ANALYSIS_DIMENSION)
        answer_text = answer_image_query(opt_img, question, api_key)
        return {
            "success": True,
            "question": question.strip(),
            "answer": answer_text
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=f"Question processing failed: {str(e)}")

@app.post("/api/agent/research", response_model=AgentResearchResponse)
async def research_agent_endpoint(request: AgentResearchRequest):
    api_key = get_api_key()
    try:
        res = execute_agent_research(request, api_key)
        return res
    except Exception as e:
        logger.error(f"Error during research agent execution: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research agent failed: {str(e)}"
        )

@app.post("/api/export")
async def export_image_endpoint(
    image_base64: str = Form(...),
    format_type: str = Form("JPEG")
):
    try:
        pil_img = base64_to_pil(image_base64)
        fmt_upper = format_type.upper()
        raw_bytes = export_image_bytes(pil_img, fmt_upper)
        
        media_type = "image/jpeg"
        filename = "image_export.jpg"
        if fmt_upper == "PNG":
            media_type = "image/png"
            filename = "image_export.png"
        elif fmt_upper == "PDF":
            media_type = "application/pdf"
            filename = "image_export.pdf"
            
        return Response(
            content=raw_bytes,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Export failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.server:app", host="0.0.0.0", port=8000, reload=True)

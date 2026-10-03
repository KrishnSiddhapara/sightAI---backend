import os
import io
import json
import base64
import logging
from typing import Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Response, status
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse
from PIL import Image

# Import existing core Python AI service modules & utilities
from utils.image_validation import validate_image_file, compute_image_hash
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

# Enable CORS for frontend integration (React Vite dev server at http://localhost:5173)
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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
    content_bytes = await file.read()
    filename = file.filename or "uploaded_image.jpg"
    pil_img = parse_and_validate_file(filename, content_bytes)
    api_key = get_api_key()

    # Safety Gate Pre-screening
    safety_result = check_image_safety(pil_img, api_key)
    if not safety_result.get("is_safe", False):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "success": False,
                "error": safety_result.get("error") or "Image rejected by safety screening gate.",
                "safety": safety_result
            }
        )

    try:
        grounded_result: GroundedAnalysisResult = analyze_image_grounded(pil_img, api_key)
        return {
            "success": True,
            "image_hash": compute_image_hash(content_bytes),
            "safety": safety_result,
            "data": grounded_result.model_dump()
        }
    except Exception as e:
        logger.error(f"Error during analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
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
        answer_text = answer_image_query(pil_img, question, api_key)
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

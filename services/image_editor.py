import io
import json
import logging
import socket
import time
from typing import Optional, Tuple, List
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
# pyrefly: ignore [missing-import]
from google.genai.errors import APIError

from services.schemas import GroundedAnalysisResult
from utils.image_validation import encode_vlm_image_part

logger = logging.getLogger(__name__)

# List of supported image-editing models in Developer API mode
PRIMARY_IMAGE_EDIT_MODEL = "gemini-2.5-flash-image"
FALLBACK_IMAGE_EDIT_MODELS = ["gemini-2.0-flash-exp", "imagen-3.0-generate-002", "gemini-2.5-flash"]

IMAGE_EDIT_SYSTEM_PROMPT = """You are an expert AI Image Editor.
Your task is to modify the input image strictly according to the user's edit request while preserving all unrelated content.

STRICT EDITING RULES:
1. REQUESTED CHANGE ONLY: Modify ONLY what the user explicitly requested.
2. PRESERVE UNRELATED CONTENT: Keep all unrelated people, objects, background details, perspective, composition, and shadows intact.
3. VISUAL COHERENCE: Ensure the edit blends naturally into the image with matching lighting, depth, and realistic texture.
4. IDENTITY PRESERVATION: Do not alter the features, face, hair, or appearance of people unless explicitly requested.
5. NO EXTRA OBJECTS: Do not add unrequested objects, text, logos, or watermarks.
"""

def validate_edit_instruction(instruction: str, max_length: int = 300) -> Optional[str]:
    """
    Validates user edit instruction string.
    Returns an error message if invalid, or None if valid.
    """
    if not instruction or not instruction.strip():
        return "Please enter an edit instruction (e.g. 'Add a football next to the person')."
    if len(instruction.strip()) > max_length:
        return f"Edit instruction is too long (maximum {max_length} characters allowed)."
    return None

def check_edit_instruction_ambiguity(
    instruction: str,
    vision_context: Optional[GroundedAnalysisResult] = None
) -> Tuple[bool, Optional[str]]:
    """
    Analyzes whether an edit instruction is ambiguous based on available vision context.
    For example: "Change the shirt to blue" when 3 distinct people exist without specifying which person.
    
    Returns:
        (is_ambiguous, clarification_message)
    """
    if not vision_context or not vision_context.objects:
        return False, None

    instr_lower = instruction.strip().lower()

    # Check person clothing target ambiguity
    person_keywords = ["shirt", "t-shirt", "jacket", "pants", "dress", "top", "clothes", "clothing", "hat", "glasses"]
    mentions_clothing = any(kw in instr_lower for kw in person_keywords)
    mentions_specific_person = any(
        kw in instr_lower for kw in [
            "person 1", "person 2", "person 3", "person 4", "person 5",
            "first person", "second person", "third person", "person on the left",
            "person on the right", "man on the left", "woman on the right",
            "guy in red", "girl in blue", "each person", "everyone", "all people"
        ]
    )

    if mentions_clothing and not mentions_specific_person:
        # Check how many people were detected
        person_cats = [cat for cat in vision_context.objects if cat.name.lower() == "person"]
        total_people = sum(cat.confirmed_count for cat in person_cats)
        if total_people > 1:
            return (
                True,
                f"Ambiguous target: The image contains {total_people} people. Please specify which person to modify (e.g., 'Change person 1\'s shirt to blue' or 'Change the shirt of the person on the left to blue')."
            )

    return False, None

def build_editing_prompt(
    instruction: str,
    vision_context: Optional[GroundedAnalysisResult] = None
) -> str:
    """
    Constructs a conceptual, highly-guided prompt for the Gemini image generation/editing model.
    Includes context from grounded vision analysis when available to maximize consistency.
    """
    prompt_parts = [
        f"USER EDIT REQUEST: {instruction.strip()}",
        "",
        "CONCEPTUAL EDIT INSTRUCTIONS:",
        f"- Perform the following edit: {instruction.strip()}",
        "- Preserve all unrelated elements, original subject identities, lighting, camera angle, and background details.",
        "- Make the change visually coherent and natural-looking.",
    ]

    if vision_context and vision_context.overall_summary:
        prompt_parts.extend([
            "",
            f"SCENE CONTEXT (For reference only): {vision_context.overall_summary}",
        ])

    return "\n".join(prompt_parts)

def edit_image(
    image: Image.Image,
    instruction: str,
    api_key: str,
    vision_context: Optional[GroundedAnalysisResult] = None,
    max_retries: int = 2,
) -> Image.Image:
    """
    Performs generative AI image editing on a PIL Image using Google GenAI SDK.
    Optimizes input image adaptively to stay strictly below 800 KB (1024 KB limit).
    """
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("API key is missing or invalid.")

    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image input object.")

    val_err = validate_edit_instruction(instruction)
    if val_err:
        raise ValueError(val_err)

    # Check target ambiguity if vision context is provided
    is_ambiguous, ambiguity_msg = check_edit_instruction_ambiguity(instruction, vision_context)
    if is_ambiguous and ambiguity_msg:
        raise ValueError(ambiguity_msg)

    # Step 1: Optimize source image to strictly remain below 6144KB (6 MB limit)
    t0_opt = time.perf_counter()
    orig_w, orig_h = image.size
    img_bytes, mime_type, opt_w, opt_h = encode_vlm_image_part(image, max_dim=2048, max_bytes=6144 * 1024)
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
    t_opt = time.perf_counter() - t0_opt

    logger.info(
        f"[IMAGE_EDIT_OPT] Original: {orig_w}x{orig_h} -> Optimized: {opt_w}x{opt_h} | "
        f"Byte size: {len(img_bytes)/1024:.1f} KB | MIME: {mime_type} (prep: {t_opt:.3f}s)"
    )

    client = genai.Client(api_key=api_key)
    full_prompt = build_editing_prompt(instruction, vision_context)

    config = types.GenerateContentConfig(
        system_instruction=IMAGE_EDIT_SYSTEM_PROMPT,
        temperature=0.3,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    models_to_try = [PRIMARY_IMAGE_EDIT_MODEL] + FALLBACK_IMAGE_EDIT_MODELS
    last_exception = None

    for model_name in models_to_try:
        for attempt in range(1, max_retries + 1):
            try:
                t0_model = time.perf_counter()
                logger.info(f"Attempting image edit with model '{model_name}' (Attempt {attempt}/{max_retries})...")
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=[image_part, full_prompt],
                    config=config,
                )

                t_edit_dur = time.perf_counter() - t0_model

                if not response or not response.candidates:
                    raise ValueError(f"No response candidates returned from model '{model_name}'.")

                candidate = response.candidates[0]
                if not candidate.content or not candidate.content.parts:
                    finish_reason = getattr(candidate, "finish_reason", "UNKNOWN")
                    raise ValueError(f"Empty content from image editor (Finish reason: {finish_reason}).")

                # Look for inline_data containing image bytes
                image_part = None
                for part in candidate.content.parts:
                    if getattr(part, "inline_data", None) and part.inline_data.data:
                        image_part = part
                        break

                if not image_part:
                    # If model returned text instead of an image
                    text_parts = [part.text for part in candidate.content.parts if getattr(part, "text", None)]
                    text_msg = " ".join(text_parts).strip() if text_parts else "No image output part returned."
                    raise ValueError(f"Model did not return a generated image: {text_msg}")

                # Decode bytes into PIL Image
                image_bytes = image_part.inline_data.data
                edited_pil = Image.open(io.BytesIO(image_bytes))
                edited_pil.load()

                # Ensure RGB mode
                if edited_pil.mode != "RGB":
                    edited_pil = edited_pil.convert("RGB")

                # Validate dimensions and non-empty status
                width, height = edited_pil.size
                if width <= 0 or height <= 0:
                    raise ValueError("Generated image has invalid zero dimensions.")

                logger.info(f"Image edit succeeded with '{model_name}': size={edited_pil.size}, mode={edited_pil.mode}")
                return edited_pil

            except (socket.gaierror, ConnectionError, TimeoutError) as e:
                last_exception = e
                logger.warning(f"Network glitch with model '{model_name}' on attempt {attempt}: {e}")
                if attempt < max_retries:
                    time.sleep(1.5)
                    continue
                break

            except APIError as e:
                last_exception = e
                err_lower = str(e).lower()
                logger.error(f"API Error with model '{model_name}': {e}")
                # If model not found or unsupported, skip retries for this model and try next model
                if "not found" in err_lower or "404" in err_lower or "unsupported" in err_lower:
                    break
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                break

            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error with model '{model_name}': {e}")
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                break

    # Translate exception into user-understandable message
    if last_exception:
        err_str = str(last_exception)
        err_lower = err_str.lower()
        if "getaddrinfo" in err_lower or "name resolution" in err_lower or isinstance(last_exception, socket.gaierror):
            raise ConnectionError(
                "Network connection failed: Unable to reach Gemini API. Please check your internet connection."
            )
        elif "quota" in err_lower or "rate limit" in err_lower or "429" in err_lower:
            raise RuntimeError(
                "API quota or rate limit exceeded. Please wait a moment before trying your edit again."
            )
        elif isinstance(last_exception, APIError):
            raise ValueError(f"Image Editing API request failed: {last_exception.message}")
        else:
            raise ValueError(f"Unable to complete image edit: {err_str}")

    raise RuntimeError("Failed to generate edited image.")

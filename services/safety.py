import json
import logging
import time
import socket
from enum import Enum
from typing import Dict, Any, Optional
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
# pyrefly: ignore [missing-import]
from google.genai.errors import APIError

from utils.json_utils import clean_json_text
from utils.image_validation import encode_vlm_image_part

logger = logging.getLogger(__name__)

class SafetyCategory(str, Enum):
    SAFE = "SAFE"
    NUDITY = "NUDITY"
    SEXUAL_CONTENT = "SEXUAL_CONTENT"
    EXPLICIT_CONTENT = "EXPLICIT_CONTENT"
    GRAPHIC_VIOLENCE = "GRAPHIC_VIOLENCE"
    VIOLENCE_AND_CIVIL_UNREST = "VIOLENCE_AND_CIVIL_UNREST"
    OTHER_SENSITIVE_CONTENT = "OTHER_SENSITIVE_CONTENT"
    UNKNOWN = "UNKNOWN"

class SafetyEvaluation(BaseModel):
    is_safe: bool = Field(
        description="Set to True ONLY if the image is completely benign and suitable for general viewing. Set to False if it contains nudity, sexual content, violence, riots, weapons, physical conflict, gore, or sensitive content."
    )
    category: SafetyCategory = Field(
        description="Primary classification category: SAFE, NUDITY, SEXUAL_CONTENT, EXPLICIT_CONTENT, GRAPHIC_VIOLENCE, VIOLENCE_AND_CIVIL_UNREST, OTHER_SENSITIVE_CONTENT, or UNKNOWN."
    )
    confidence: float = Field(
        description="Confidence score of classification between 0.0 and 1.0."
    )
    reasoning: str = Field(
        description="Short internal rationale for safety determination."
    )

SAFETY_SYSTEM_PROMPT = """You are a strict Image Content Safety & Moderation Classifier.
Your task is to analyze the provided image and classify whether it is SAFE or UNSAFE for a general public application.

Strictly inspect for the following UNSAFE categories:
1. VIOLENCE & CIVIL UNREST: Any active violence, street violence, riots, mob clashes, physical altercations, tear gas, smoke bombs in conflict, individuals brandishing weapons, sticks, batons or dangerous objects in conflict scenes, assault, or hostile civil unrest.
2. GRAPHIC VIOLENCE / GORE: Severe open wounds, exposed organs, dismemberment, extreme gore, graphic violent scenes, bloodshed.
3. NUDITY / EXPLICIT SEXUAL CONTENT: Fully nude people, explicit nudity, visible genitals, pornographic imagery, explicit sexual acts, sexually explicit photos.
4. SEXUALIZED / ADULT CONTENT: Highly sexualized imagery, explicit sexual poses, images primarily intended for sexual content.
5. OTHER HIGHLY SENSITIVE CONTENT: Images clearly inappropriate for general-purpose public applications.

STRICT EVALUATION RULES:
- You MUST classify any image depicting active violence, riots, physical clashes, mob altercations, people brandishing weapons/sticks in conflict, tear gas clashes, assault, or graphic gore as is_safe=False and category as VIOLENCE_AND_CIVIL_UNREST or GRAPHIC_VIOLENCE.
- Do NOT block peaceful everyday activities, sports events, benign portraits, standard scenery, interior decor, furniture, household items, antique objects, retro appliances, lamps, plants, crafts, or non-violent public gatherings.
- Everyday objects, home decor, furniture, craft items, plants, tools, vintage props, nature, splash art, and household items MUST be classified as is_safe=True and category=SAFE.
- If an image depicts violence, aggression, weapons in conflict, or hostile civil unrest, you MUST mark is_safe=False.
"""

def get_fail_closed_response(error_detail: str = "Safety check unavailable") -> Dict[str, Any]:
    """
    Returns a fail-safe (fail-closed) safety result indicating the image cannot be confirmed safe.
    """
    return {
        "is_safe": False,
        "category": SafetyCategory.UNKNOWN.value,
        "confidence": 0.0,
        "reasoning": f"Fail-closed guard triggered: {error_detail}",
        "error": f"⚠️ Safety check unavailable: {error_detail}",
    }

def check_image_safety(image: Image.Image, api_key: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    Performs pre-analysis safety screening on an image before normal processing.
    Includes automatic retries for network/DNS glitches and configures BLOCK_NONE
    on the classification call so Gemini outputs the structured SafetyEvaluation JSON.
    
    Returns a dictionary with keys:
        - is_safe (bool)
        - category (str)
        - confidence (float)
        - reasoning (str)
        - error (Optional[str])
    """
    if not api_key or api_key == "your_api_key_here":
        return get_fail_closed_response("API key is missing or unconfigured.")

    if not isinstance(image, Image.Image):
        return get_fail_closed_response("Invalid image input object.")

    client = genai.Client(api_key=api_key)

    # Use BLOCK_NONE on the classification call itself so Gemini will NOT block
    # its own output text when returning the SafetyEvaluation JSON structure.
    native_safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    config = types.GenerateContentConfig(
        system_instruction=SAFETY_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=SafetyEvaluation,
        safety_settings=native_safety_settings,
        temperature=0.0,
        max_output_tokens=512,
    )

    last_error_detail = None
    t0 = time.perf_counter()

    img_bytes, mime_type, _, _ = encode_vlm_image_part(image)
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_part, "Perform strict safety screening on this image."],
                config=config,
            )

            # Check if response was empty
            if not response or not response.text:
                finish_reason = None
                if response and response.candidates:
                    finish_reason = getattr(response.candidates[0], "finish_reason", None)
                
                logger.warning(f"Safety API response empty on attempt {attempt}. Finish reason: {finish_reason}")
                last_error_detail = f"Content safety screening failed (Finish reason: {finish_reason})."
                if attempt < max_retries:
                    time.sleep(1.0)
                    continue
                break

            # Parse structured response
            raw_text = clean_json_text(response.text)
            data = json.loads(raw_text)
            
            is_safe = bool(data.get("is_safe", False))
            category = str(data.get("category", SafetyCategory.UNKNOWN.value)).upper()
            confidence = float(data.get("confidence", 0.0))
            reasoning = str(data.get("reasoning", ""))

            # Ensure category aligns with known SafetyCategory values
            valid_categories = {c.value for c in SafetyCategory}
            if category not in valid_categories:
                category = SafetyCategory.OTHER_SENSITIVE_CONTENT.value if not is_safe else SafetyCategory.SAFE.value

            # Enforce consistency: if category is not SAFE, is_safe MUST be False
            if category != SafetyCategory.SAFE.value:
                is_safe = False

            return {
                "is_safe": is_safe,
                "category": category,
                "confidence": confidence,
                "reasoning": reasoning,
                "error": None if is_safe else f"⚠️ Content flagged: Image was categorized as {category}. {reasoning}",
            }

        except (socket.gaierror, ConnectionError, TimeoutError) as e:
            logger.warning(f"Network glitch during safety check (attempt {attempt}/{max_retries}): {e}")
            last_error_detail = "Network connection failed during safety verification."
            if attempt < max_retries:
                time.sleep(1.5)
                continue
            break

        except APIError as e:
            logger.error(f"Gemini API error during safety check: {e}")
            last_error_detail = f"Gemini API service error: {e.message}"
            break

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse safety evaluation JSON (attempt {attempt}/{max_retries}): {e}. Raw text: {response.text[:200] if response and response.text else 'Empty'}")
            last_error_detail = "Invalid JSON safety response format."
            
            # Fallback string matching if JSON parsing fails but explicit SAFE string is present
            if response and response.text and '"is_safe": true' in response.text.lower():
                logger.info("Recovered safety check result via fallback string matching (is_safe=True).")
                return {
                    "is_safe": True,
                    "category": SafetyCategory.SAFE.value,
                    "confidence": 0.9,
                    "reasoning": "Recovered via fallback string parsing.",
                    "error": None
                }

            if attempt < max_retries:
                time.sleep(1.0)
                continue
            break

        except Exception as e:
            logger.error(f"Unexpected exception during safety check: {e}")
            last_error_detail = f"Safety classifier error: {str(e)}"
            if attempt < max_retries:
                time.sleep(1.0)
                continue
            break

    return get_fail_closed_response(last_error_detail or "We could not verify this image safely. Please try again.")

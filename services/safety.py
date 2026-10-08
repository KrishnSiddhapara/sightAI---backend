import json
import logging
import time
import socket
from enum import Enum
from typing import Dict, Any, Optional, Union
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import APIError

from utils.json_utils import clean_json_text
from utils.image_validation import encode_vlm_image_part
from utils.config import PRIMARY_VLM_MODEL, FALLBACK_VLM_MODEL, ANALYSIS_TIMEOUT

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

def get_fail_closed_response(error_detail: str = "Safety check unavailable", error_code: str = "SAFETY_GATE_ERROR") -> Dict[str, Any]:
    """
    Returns a fail-safe (fail-closed) safety result indicating the image cannot be confirmed safe.
    """
    return {
        "is_safe": False,
        "category": SafetyCategory.UNKNOWN.value,
        "confidence": 0.0,
        "reasoning": f"Fail-closed guard triggered: {error_detail}",
        "error": f"⚠️ Safety check unavailable: {error_detail}",
        "error_code": error_code,
    }

def check_image_safety(image: Union[Image.Image, types.Part], api_key: str) -> Dict[str, Any]:
    """
    Performs pre-analysis safety screening on an image or pre-encoded Part.
    Optimized to use pre-encoded Part, explicit timeouts, and controlled fast fallback.
    
    Returns a dictionary with keys:
        - is_safe (bool)
        - category (str)
        - confidence (float)
        - reasoning (str)
        - error (Optional[str])
        - error_code (Optional[str])
    """
    if not api_key or api_key == "your_api_key_here":
        return get_fail_closed_response("API key is missing or unconfigured.", error_code="API_KEY_ERROR")

    if isinstance(image, types.Part):
        image_part = image
    elif isinstance(image, Image.Image):
        img_bytes, mime_type, _, _ = encode_vlm_image_part(image)
        image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
    else:
        return get_fail_closed_response("Invalid image input object.", error_code="INVALID_IMAGE")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=int(ANALYSIS_TIMEOUT * 1000))
    )

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
        max_output_tokens=1024,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    models_to_try = ["gemini-2.5-flash"]

    last_error_detail = None

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[image_part, "Perform strict safety screening on this image."],
                config=config,
            )

            if not response or not response.text:
                last_error_detail = f"Safety response empty on model '{model_name}'."
                continue

            raw_text = clean_json_text(response.text)
            data = None
            try:
                data = json.loads(raw_text)
            except Exception as json_err:
                first_b = raw_text.find('{')
                last_b = raw_text.rfind('}')
                if first_b != -1 and last_b != -1 and first_b < last_b:
                    try:
                        data = json.loads(raw_text[first_b:last_b+1])
                    except Exception:
                        data = None

            if not isinstance(data, dict):
                last_error_detail = f"Malformed safety JSON output: {raw_text[:200]}"
                continue

            is_safe = bool(data.get("is_safe", False))
            category = str(data.get("category", SafetyCategory.UNKNOWN.value)).upper()
            confidence = float(data.get("confidence", 0.0))
            reasoning = str(data.get("reasoning", ""))

            valid_categories = {c.value for c in SafetyCategory}
            if category not in valid_categories:
                category = SafetyCategory.OTHER_SENSITIVE_CONTENT.value if not is_safe else SafetyCategory.SAFE.value

            if category != SafetyCategory.SAFE.value:
                is_safe = False

            logger.info(f"[SAFETY_SUCCESS] Model='{model_name}': is_safe={is_safe}, category={category}")
            return {
                "is_safe": is_safe,
                "category": category,
                "confidence": confidence,
                "reasoning": reasoning,
                "error": None if is_safe else f"⚠️ Content flagged: Image categorized as {category}. {reasoning}",
                "error_code": None if is_safe else "SAFETY_FLAGGED"
            }

        except Exception as e:
            last_error_detail = str(e)
            logger.warning(f"[SAFETY_WARN] Safety check exception on model '{model_name}': {e}")
            continue

    return get_fail_closed_response(last_error_detail or "Safety screening could not verify image safety.")

PROHIBITED_EDIT_PATTERNS = [
    # Nudity & Sexual Content
    (r"\b(naked|nude|nudity|undress|undressed|topless|bottomless|porn|porno|pornography|sex|sexual|sexually|erotic|genitals|breast|breasts|penis|vagina|lingerie|strip)\b", SafetyCategory.NUDITY, "Nudity or sexually explicit content is prohibited."),
    # Graphic Violence & Gore
    (r"\b(blood|bloody|bleed|bleeding|gore|gory|stab|stabbed|stabbing|shoot|shooting|shot|kill|killed|killing|murder|murdered|decapitate|decapitated|mutilate|mutilated|wound|wounded|open wound|bullet hole|corpse|slashing|slashed)\b", SafetyCategory.GRAPHIC_VIOLENCE, "Graphic violence, gore, or physical harm is prohibited."),
    # Violence, Civil Unrest & Weapons Assault
    (r"\b(assault|assaulting|beating|attack|attacking|gunfire|terrorist|terrorism|bomb|bombing|explosion|decapitation)\b", SafetyCategory.VIOLENCE_AND_CIVIL_UNREST, "Violence, attacks, terrorism, or dangerous conflict imagery is prohibited."),
    # Self-Harm & Hate Speech
    (r"\b(suicide|self-harm|self harm|hanging|nazi|swastika)\b", SafetyCategory.OTHER_SENSITIVE_CONTENT, "Offensive, hate speech, or sensitive content is prohibited.")
]

def check_edit_prompt_safety(instruction: str, api_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Performs safety guardrail screening on user image edit text instruction.
    Returns structured safety dictionary with is_safe, category, reasoning, and error.
    """
    import re
    if not instruction or not instruction.strip():
        return {
            "is_safe": True,
            "category": SafetyCategory.SAFE.value,
            "confidence": 1.0,
            "reasoning": "Empty instruction.",
            "error": None,
            "error_code": None
        }

    instr_lower = instruction.strip().lower()

    # Rule 1: Fast pattern matching against prohibited safety categories
    for pattern, category, desc in PROHIBITED_EDIT_PATTERNS:
        if re.search(pattern, instr_lower):
            logger.warning(f"[SAFETY_PROMPT_BLOCK] Edit instruction blocked by pattern '{pattern}': category={category.value}")
            return {
                "is_safe": False,
                "category": category.value,
                "confidence": 0.98,
                "reasoning": f"Edit instruction violates safety guardrails policy: {desc}",
                "error": f"⚠️ Safety Guardrail Triggered: Requested edit contains prohibited content ({category.value.replace('_', ' ')}). {desc}",
                "error_code": "PROMPT_SAFETY_FLAGGED"
            }

    return {
        "is_safe": True,
        "category": SafetyCategory.SAFE.value,
        "confidence": 1.0,
        "reasoning": "Instruction passed safety guardrail pre-screening.",
        "error": None,
        "error_code": None
    }


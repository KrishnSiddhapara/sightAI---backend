import json
import logging
import time
import socket
from typing import List, Tuple, Optional
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
# pyrefly: ignore [missing-import]
from google.genai.errors import APIError

from services.schemas import GroundedAnalysisResult

logger = logging.getLogger(__name__)

GROUNDED_VISION_SYSTEM_PROMPT = """You are a world-class Computer Vision & Physical Object Verification AI.
Your absolute top priority is VISUAL ACCURACY, GROUNDING, and ZERO HALLUCINATION.

Follow this MANDATORY step-by-step visual verification pipeline before producing structured output:

STEP 1: FULL IMAGE SCAN & ENTITY DETECTION
- Scan the entire image for distinct, visually observable physical entities.
- Distinguish between real physical objects in the 3D scene vs reflections in mirrors/glass, shadows on surfaces, or pictures shown inside posters/TV screens/paintings.
- DO NOT count reflections, shadows, or images shown inside screens/posters/photos as real physical objects.

STEP 2: PHYSICAL INSTANCE COUNTING & DEDUPLICATION
- Count distinct physical instances.
- DO NOT double-count an object under different names (e.g. if an item is a 'car', do NOT list it as 'car' and also as 'vehicle' or 'automobile').
- Standardize object category names to simple lowercase terms (e.g. 'person', 'car', 'dog', 'chair', 'bottle', 'bicycle').
- For overlapping or partially hidden objects:
  * Count ONLY clearly distinguishable physical instances under 'confirmed_count'.
  * If an object is partially visible but cannot be confirmed, increment 'uncertain_count' and explain the reason in 'uncertainty_reason'. Do NOT guess or pad the confirmed count.

STEP 3: INDEPENDENT PERSON-BY-PERSON & INSTANCE ATTRIBUTE EXTRACTION
- EVERY person instance (person_1, person_2, person_3, etc.) MUST be analyzed independently based strictly on their visible spatial region.
- NEVER create a generic description and copy it to multiple people.
- If Person 1 wears a red shirt and Person 2 wears a blue jacket, Person 1 MUST have clothing_color='red' and Person 2 MUST have clothing_color='blue'. NEVER share or copy attributes across entities!
- If an attribute (e.g. exact clothing, pose, action, age, brand) is not clearly visible, return "not clearly visible" or "unknown". DO NOT guess!

STEP 4: IMAGE-WIDE CONSISTENCY CROSS-CHECK
Before returning the final structured JSON, cross-check:
A. Is every reported object actually visible in the image?
B. Is the confirmed count equal to the number of distinct physical instances?
C. Are attributes assigned strictly to their correct owner entity?
D. Were any shadows, reflections, or photos inside screens mistakenly counted?
E. Are minor background details excluded unless clearly prominent?
"""

def analyze_image_grounded(image: Image.Image, api_key: str, max_retries: int = 3) -> GroundedAnalysisResult:
    """
    Sends PIL image to Gemini 2.5 Flash with structured Pydantic schema (GroundedAnalysisResult).
    Includes automatic retries for transient network glitches.
    """
    if not api_key:
        raise ValueError("API key is missing.")

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=GROUNDED_VISION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=GroundedAnalysisResult,
        temperature=0.0,  # Deterministic temperature for maximum precision & zero hallucination
    )

    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image, "Perform strict step-by-step visual verification, physical instance counting, independent attribute analysis, and scene understanding on this image."],
                config=config,
            )

            if not response or not response.text:
                raise ValueError("Received an empty response from Gemini Vision API.")

            data = json.loads(response.text)
            return GroundedAnalysisResult.model_validate(data)

        except (socket.gaierror, ConnectionError, TimeoutError, APIError, Exception) as e:
            last_exception = e
            err_str = str(e).lower()
            is_network_err = (
                isinstance(e, (socket.gaierror, ConnectionError, TimeoutError))
                or "getaddrinfo" in err_str
                or "connection" in err_str
            )
            if is_network_err and attempt < max_retries:
                logger.warning(f"Network glitch on attempt {attempt}/{max_retries}: {e}. Retrying in 1.5s...")
                time.sleep(1.5)
                continue
            else:
                break

    if last_exception:
        err_msg = str(last_exception)
        if "getaddrinfo" in err_msg.lower() or isinstance(last_exception, socket.gaierror):
            raise ConnectionError("Network connection failed: DNS lookup failed. Please check your internet connection.")
        elif isinstance(last_exception, APIError):
            raise ValueError(f"Gemini API request failed: {last_exception.message}")
        else:
            raise ValueError(f"Failed to perform visual analysis: {err_msg}")

    raise ValueError("Unexpected error during visual analysis.")

def parse_vlm_response(response_text: str) -> Tuple[List[str], str]:
    """
    Legacy parser function maintained for backwards compatibility.
    """
    objects_list: List[str] = []
    description: str = ""
    try:
        lines = [line.strip() for line in response_text.strip().split("\n")]
        current_section = None
        for line in lines:
            if not line:
                continue
            lower_line = line.lower()
            if "objects:" in lower_line or "objects detected:" in lower_line:
                current_section = "objects"
                continue
            elif "description:" in lower_line:
                current_section = "description"
                continue
            if current_section == "objects":
                clean_line = line
                if line.startswith("-") or line.startswith("*"):
                    clean_line = line[1:].strip()
                elif "." in line:
                    parts = line.split(".", 1)
                    if parts[0].strip().isdigit():
                        clean_line = parts[1].strip()
                objects_list.append(clean_line)
            elif current_section == "description":
                description += line + " "
        description = description.strip()
    except Exception as e:
        logger.warning(f"Error while parsing legacy VLM response: {e}")
    return objects_list, description

def analyze_image(image: Image.Image, api_key: str) -> str:
    """
    Legacy plain-text analysis function maintained for backwards compatibility.
    """
    if not api_key:
        raise ValueError("API key is missing.")

    client = genai.Client(api_key=api_key)
    prompt = """Analyze this image and identify the important visible objects. Return Objects list and Description."""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[image, prompt]
    )

    if not response or not response.text:
        raise ValueError("Received an empty response from the VLM API.")

    return response.text

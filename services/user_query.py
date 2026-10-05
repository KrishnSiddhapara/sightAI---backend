import logging
import time
import socket
from typing import Optional
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types
# pyrefly: ignore [missing-import]
from google.genai.errors import APIError
from utils.image_validation import prepare_ask_ai_image_part

logger = logging.getLogger(__name__)

USER_QUERY_SYSTEM_PROMPT = """You are SightAI's Advanced AI Visual & Technical Assistant.

CORE GUIDELINES:
1. ADAPTIVE ANSWER DEPTH:
   - Adapt answer depth intelligently based on question complexity.
   - For simple visual facts (e.g., "What color is the shirt?"), answer directly and clearly.
   - For complex, conceptual, technical, or analytical questions about the image (e.g. "Explain the architecture of the server in this photo", "How does this device work?"), provide a comprehensive, well-structured, detailed response with headings, bullet points, examples, and practical explanations.
2. STRICT GROUNDING:
   - Ground every observation in the visual evidence provided by the image and existing vision analysis.
   - If an element cannot be reliably determined, state it clearly rather than guessing.
3. RICH MARKDOWN FORMATTING:
   - Use Markdown headers (`##`, `###`), lists, bold emphasis, code blocks, and tables to present information cleanly.
"""

def validate_user_prompt(prompt: str, max_length: int = 500) -> Optional[str]:
    """
    Validates user query string. Returns an error message if invalid, or None if valid.
    """
    if not prompt or not prompt.strip():
        return "Please enter a question about the image."
    if len(prompt.strip()) > max_length:
        return f"Question is too long (maximum {max_length} characters allowed)."
    return None

def answer_image_query(image: Image.Image, user_prompt: str, api_key: str, max_retries: int = 3) -> str:
    """
    Answers a custom question about a single safety-approved image using Gemini 2.5 Flash.
    Includes automatic retries and user-friendly error translation for network/DNS failures.
    """
    if not api_key:
        raise ValueError("API key is missing.")

    err = validate_user_prompt(user_prompt)
    if err:
        raise ValueError(err)

    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        system_instruction=USER_QUERY_SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=4096,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    img_bytes, mime_type, _, _ = prepare_ask_ai_image_part(image)
    image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)

    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[image_part, f"User Question: {user_prompt.strip()}"],
                config=config,
            )

            if not response or not response.text:
                raise ValueError("Received an empty response from the AI assistant.")

            return response.text.strip()

        except (socket.gaierror, ConnectionError, TimeoutError, APIError, Exception) as e:
            last_exception = e
            err_str = str(e).lower()

            # Check if network / DNS failure
            is_network_err = (
                isinstance(e, (socket.gaierror, ConnectionError, TimeoutError))
                or "getaddrinfo" in err_str
                or "name resolution" in err_str
                or "connection" in err_str
            )

            if is_network_err and attempt < max_retries:
                logger.warning(f"Network glitch on attempt {attempt}/{max_retries}: {e}. Retrying in 1.5s...")
                time.sleep(1.5)
                continue
            else:
                break

    # If all attempts failed, raise human-readable error message
    if last_exception:
        err_msg = str(last_exception)
        err_lower = err_msg.lower()

        if "getaddrinfo" in err_lower or "name resolution" in err_lower or isinstance(last_exception, socket.gaierror):
            raise ConnectionError(
                "Network connection failed: Unable to resolve Google API host (DNS lookup failed). "
                "Please verify your internet connection and try asking again."
            )
        elif "connection" in err_lower or isinstance(last_exception, ConnectionError):
            raise ConnectionError(
                "Connection error: Could not reach Google Gemini servers. Please check your network and try again."
            )
        elif isinstance(last_exception, APIError):
            raise ValueError(f"Gemini API request failed: {last_exception.message}")
        else:
            raise RuntimeError(f"Unable to process question: {err_msg}")

    raise RuntimeError("Unexpected error occurred while processing question.")

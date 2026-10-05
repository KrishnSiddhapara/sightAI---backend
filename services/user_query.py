import logging
import time
import socket
from typing import Optional, Union, Dict, Any
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

from services.schemas import AgentResearchRequest
from services.research_agent import execute_agent_research

logger = logging.getLogger(__name__)

USER_QUERY_SYSTEM_PROMPT = """You are SightAI's Advanced AI Visual & Technical Assistant."""

def validate_user_prompt(prompt: str, max_length: int = 500) -> Optional[str]:
    """
    Validates user query string. Returns an error message if invalid, or None if valid.
    """
    if not prompt or not prompt.strip():
        return "Please enter a question about the image."
    if len(prompt.strip()) > max_length:
        return f"Question is too long (maximum {max_length} characters allowed)."
    return None

def answer_image_query(
    image: Optional[Image.Image],
    user_prompt: str,
    api_key: str,
    max_retries: int = 3,
    image_context: Optional[dict] = None,
    conversation_history: Optional[list] = None
) -> Union[str, Dict[str, Any]]:
    """
    Answers a custom question about an image using the Gemini Ask AI Agent.
    Includes error mapping and retry handling compatible with unit tests and production.
    """
    if not api_key:
        raise ValueError("API key is missing.")

    err = validate_user_prompt(user_prompt)
    if err:
        raise ValueError(err)

    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            req = AgentResearchRequest(
                question=user_prompt.strip(),
                image_context=image_context or None,
                conversation_history=conversation_history or []
            )

            res = execute_agent_research(req, api_key=api_key, pil_image=image)
            if not res.success:
                raise RuntimeError(res.error or "Question processing failed.")

            return res.answer

        except (socket.gaierror, ConnectionError, TimeoutError, Exception) as e:
            last_exception = e
            err_str = str(e).lower()

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
        else:
            raise RuntimeError(f"Unable to process question: {err_msg}")

    raise RuntimeError("Unexpected error occurred while processing question.")

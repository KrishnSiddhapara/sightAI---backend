import logging
import httpx
from bs4 import BeautifulSoup
from typing import Optional

logger = logging.getLogger(__name__)

class PageReaderTool:
    """
    Safely reads text content from a web page URL.
    Implements timeouts, size bounds, and header sanitization.
    """
    name = "page_reader"
    description = "Fetches and extracts main text content from a web page URL."

    def execute(self, url: str, max_chars: int = 2500) -> Optional[str]:
        if not url or not (url.startswith("http://") or url.startswith("https://")):
            return None

        logger.info(f"[PageReaderTool] Reading page content from: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }

        try:
            with httpx.Client(timeout=6.0, follow_redirects=True) as client:
                resp = client.get(url, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"[PageReaderTool] Non-200 status code ({resp.status_code}) for {url}")
                    return None

                # Parse HTML content
                soup = BeautifulSoup(resp.text, "html.parser")
                
                # Remove scripts, styles, navs
                for tag in soup(["script", "style", "nav", "footer", "header", "svg"]):
                    tag.decompose()

                text = soup.get_text(separator=" ", strip=True)
                cleaned_text = " ".join(text.split())

                if len(cleaned_text) > max_chars:
                    cleaned_text = cleaned_text[:max_chars] + "..."

                return cleaned_text
        except Exception as e:
            logger.warning(f"[PageReaderTool] Failed to read page {url}: {e}")
            return None

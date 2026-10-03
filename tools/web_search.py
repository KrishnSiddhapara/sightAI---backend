import logging
import urllib.parse
from typing import List, Dict, Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Known domain classification maps
RETAILER_DOMAINS = {
    'amazon.com', 'amazon.in', 'amazon.co.uk', 'amazon.ca', 'amazon.de',
    'walmart.com', 'target.com', 'ebay.com', 'bestbuy.com', 'flipkart.com',
    'bookswagon.com', 'barnesandnoble.com', 'abebooks.com', 'etsy.com'
}

REFERENCE_DOMAINS = {
    'wikipedia.org', 'en.wikipedia.org', 'britannica.com', 'goodreads.com',
    'imdb.com', 'github.com', 'arxiv.org'
}

PUBLISHER_DOMAINS = {
    'penguinrandomhouse.com', 'harpercollins.com', 'simonandschuster.com',
    'macmillan.com', 'hachettebookgroup.com', 'bloomsbury.com', 'wiley.com',
    'oreilly.com', 'packtpub.com', 'pearson.com'
}

def extract_domain(url_str: str) -> str:
    try:
        parsed = urlparse(url_str)
        domain = parsed.netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except Exception:
        return ""

def classify_source_type(url_str: str, title: str = "", snippet: str = "") -> str:
    domain = extract_domain(url_str)
    
    if any(pub in domain for pub in PUBLISHER_DOMAINS):
        return "publisher"
    if any(ret in domain for ret in RETAILER_DOMAINS):
        return "retailer"
    if any(ref in domain for ref in REFERENCE_DOMAINS):
        return "reference"
    
    # Check official hints
    lower_title = title.lower()
    lower_snippet = snippet.lower()
    if "official site" in lower_title or "official website" in lower_title or "official site" in lower_snippet:
        return "official"
        
    return "general"

class WebSearchTool:
    """
    Search Tool wrapping DDGS and fallback HTTP scraper.
    Returns structured list of Dicts: {title, url, domain, snippet, source_type}.
    """
    name = "web_search"
    description = "Searches the web for current real-world information, availability, prices, official sources, and links."

    def execute(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        logger.info(f"[WebSearchTool] Executing search query: '{query}'")
        results: List[Dict[str, Any]] = []

        # Strategy 1: ddgs
        try:
            from ddgs import DDGS
            ddgs = DDGS()
            raw_results = list(ddgs.text(query, max_results=max_results))
            for item in raw_results:
                href = item.get('href') or item.get('url') or ""
                if not href or not (href.startswith('http://') or href.startswith('https://')):
                    continue
                
                title = item.get('title') or "Web Result"
                snippet = item.get('body') or item.get('snippet') or ""
                domain = extract_domain(href)
                stype = classify_source_type(href, title, snippet)

                results.append({
                    "title": title.strip(),
                    "url": href.strip(),
                    "domain": domain,
                    "snippet": snippet.strip(),
                    "source_type": stype
                })

            if results:
                logger.info(f"[WebSearchTool] DDGS returned {len(results)} verified results.")
                return results[:max_results]

        except Exception as e:
            logger.warning(f"[WebSearchTool] DDGS search failed: {e}. Trying fallback...")

        # Strategy 2: Fallback HTML DDG search via httpx & BeautifulSoup
        try:
            import httpx
            from bs4 import BeautifulSoup

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            params = {"q": query}
            response = httpx.get("https://html.duckduckgo.com/html/", params=params, headers=headers, timeout=8.0)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                for a_tag in soup.find_all("a", class_="result__url"):
                    href = a_tag.get("href", "")
                    # Extract actual URL from DDG proxy url if needed
                    if "/l/?" in href and "uddg=" in href:
                        parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
                        if "uddg" in parsed_qs:
                            href = parsed_qs["uddg"][0]
                    
                    if href.startswith('http://') or href.startswith('https://'):
                        title_elem = a_tag.find_parent("div", class_="result__body")
                        title = title_elem.find("a", class_="result__a").text.strip() if title_elem and title_elem.find("a", class_="result__a") else "Web Source"
                        snippet_elem = title_elem.find("a", class_="result__snippet") if title_elem else None
                        snippet = snippet_elem.text.strip() if snippet_elem else ""
                        
                        domain = extract_domain(href)
                        stype = classify_source_type(href, title, snippet)
                        
                        results.append({
                            "title": title,
                            "url": href,
                            "domain": domain,
                            "snippet": snippet,
                            "source_type": stype
                        })
                        if len(results) >= max_results:
                            break
                logger.info(f"[WebSearchTool] Fallback scraper returned {len(results)} results.")
        except Exception as fallback_err:
            logger.error(f"[WebSearchTool] Fallback search error: {fallback_err}")

        return results[:max_results]

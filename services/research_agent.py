import json
import logging
import time
import base64
import io
import urllib.parse
from typing import List, Dict, Any, Optional
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
from google import genai
# pyrefly: ignore [missing-import]
from google.genai import types

from services.schemas import (
    AgentResearchRequest,
    AgentResearchResponse,
    SourceItem,
    ProductEntity,
    FactItem
)
from tools.web_search import WebSearchTool, extract_domain, classify_source_type
from tools.page_reader import PageReaderTool
from utils.json_utils import clean_json_text
from utils.image_validation import prepare_ask_ai_image_part

logger = logging.getLogger(__name__)

def decode_base64_to_pil(b64_str: str) -> Optional[Image.Image]:
    """Helper to decode a base64 image string into a PIL Image."""
    if not b64_str or not isinstance(b64_str, str):
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        image_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(image_bytes))
        img.load()
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        logger.warning(f"[ResearchAgent] Could not decode base64 image: {e}")
        return None

def build_retailer_search_link(retailer_name: str, product_query: str) -> str:
    """Builds working search URLs for top Indian marketplaces preserving exact item search terms."""
    q = urllib.parse.quote_plus(product_query.strip())
    r = retailer_name.lower()
    if 'amazon' in r:
        return f"https://www.amazon.in/s?k={q}"
    elif 'flipkart' in r:
        return f"https://www.flipkart.com/search?q={q}"
    elif 'decathlon' in r:
        return f"https://www.decathlon.in/search?query={q}"
    elif 'croma' in r:
        return f"https://www.croma.com/searchB?q={q}"
    elif 'myntra' in r:
        return f"https://www.myntra.com/{q}"
    elif 'nykaa' in r:
        return f"https://www.nykaa.com/search/result/?q={q}"
    elif 'reliance' in r:
        return f"https://www.reliancedigital.in/search?q={q}"
    elif 'tata' in r:
        return f"https://www.tatacliq.com/search/?text={q}"
    else:
        return f"https://www.google.co.in/search?tbm=shop&q={q}"

INTENT_DETECTION_SYSTEM_PROMPT = """You are SightAI's Advanced Intent Classification, Image Context & Query Planning Engine.
Examine the user's question, previous conversation history, and available visual context (image + vision analysis) to classify intent and plan required tools.

INTENT CATEGORIES:
1. IMAGE_QUESTION: Visual questions about the image (e.g., "What is this?", "What color is it?", "What objects are visible?").
2. GENERAL_QUESTION: General knowledge or conceptual questions (e.g., "What is a football?", "Define machine learning").
3. TECHNICAL_QUESTION: In-depth technical topics, engineering, architecture, systems (e.g., "How does RAG work?", "Explain WebRTC").
4. PROGRAMMING: Code, Python, JavaScript, APIs, debugging, code snippets.
5. PRODUCT_IDENTIFICATION: Identifying what specific product, device, book, item, or model is shown in the image.
6. PRODUCT_RESEARCH: Detailed research into a product's specs, capabilities, features, or background.
7. PRODUCT_PRICE: Current pricing info, deals, market rates. Set requires_research=true.
8. PRODUCT_AVAILABILITY: Retailers, stock, availability. Set requires_research=true.
9. PRODUCT_PURCHASE: "Where can I buy this?", purchase options, shopping links. Set requires_research=true.
10. PRODUCT_COMPARISON: Comparing items/products (e.g., "Compare Amazon vs Flipkart", "Compare S24 vs iPhone 15").
11. LATEST_INFORMATION: Fresh real-time info, news, current events, latest 2026 releases. Set requires_research=true.
12. NEWS_RESEARCH: News topics, recent events, breaking developments. Set requires_research=true.
13. COMPANY_RESEARCH: Manufacturer, store, brand, or company background. Set requires_research=true.
14. LOCATION_RESEARCH: Physical places, landmarks, store locations, regional availability. Set requires_research=true.
15. GENERAL_RESEARCH: Any topic requiring live web search. Set requires_research=true.
16. OTHER: Any other query.

RULES:
- MULTI-TURN CONVERSATION RESOLUTION: Resolve pronouns ("its price", "where to buy it", "that phone", "which seller is cheaper") using previous conversation turns and detected visual entities.
- REQUIRES_IMAGE: Set requires_image=true whenever the question asks about the currently uploaded image or refers to an object shown in it.
- REQUIRES_HIGH_QUALITY_IMAGE: Set requires_high_quality_image=true if the question requires inspecting fine visual details like small text, product labels, model numbers, logos, packaging, fine textures, or exact serial numbers.
- INDIA FIRST PRIORITY: Unless the user explicitly specifies another target country (e.g., "in USA", "in UK", "in Japan"), construct search queries prioritizing INDIA sellers, INR pricing (₹), and Indian marketplaces (e.g., Amazon India, Flipkart).
- PRESERVE USER KEYWORDS: Retain exact item keywords (brand, model, color, variant, size, storage) provided by user or identified in visual context. (e.g., "Samsung S25 Ultra 512GB Black India").
"""

INTENT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {
            "type": "STRING",
            "description": "One of IMAGE_QUESTION, GENERAL_QUESTION, TECHNICAL_QUESTION, PROGRAMMING, PRODUCT_IDENTIFICATION, PRODUCT_RESEARCH, PRODUCT_PRICE, PRODUCT_AVAILABILITY, PRODUCT_PURCHASE, PRODUCT_COMPARISON, LATEST_INFORMATION, NEWS_RESEARCH, COMPANY_RESEARCH, LOCATION_RESEARCH, GENERAL_RESEARCH, OTHER"
        },
        "requires_image": {"type": "BOOLEAN", "description": "True if answering needs the uploaded image context."},
        "requires_high_quality_image": {"type": "BOOLEAN", "description": "True if reading small text, labels, logos, or fine details requires high-quality image."},
        "requires_research": {"type": "BOOLEAN", "description": "True if external web search is needed for current info/pricing/sellers."},
        "is_shopping_query": {"type": "BOOLEAN", "description": "True if query asks where to buy, prices, sellers, deals, or shopping availability."},
        "user_country": {"type": "STRING", "description": "Target country if explicitly specified by user, else null."},
        "resolved_entity": {
            "type": "OBJECT",
            "properties": {
                "name": {"type": "STRING"},
                "brand": {"type": "STRING"},
                "model": {"type": "STRING"},
                "variant": {"type": "STRING"},
                "category": {"type": "STRING"}
            }
        },
        "search_query": {"type": "STRING", "description": "Targeted search query for web search prioritizing India sellers if shopping query."},
        "reasoning": {"type": "STRING"}
    },
    "required": ["intent", "requires_image", "requires_research", "is_shopping_query", "search_query"]
}

RESEARCH_SYNTHESIS_SYSTEM_PROMPT = """You are SightAI's Advanced General AI Assistant & Enterprise Agent powered by Gemini.

CORE RESPONSE POLICIES & BEHAVIOR:

1. ADAPTIVE ANSWER DEPTH:
   - Match answer depth to question complexity automatically.
   - Simple visual or trivial question ("What is this?", "What color is it?") → concise, direct response.
   - Technical, conceptual, programming, or AI/ML question ("Explain RAG", "How does FastAPI DI work?") → comprehensive, detailed explanation using Markdown headings (`##`, `###`), bold text, bullet points, and code blocks.
   - Research or shopping question → deep, structured response with product breakdown, pricing table, recommendations, and important notes.

2. INDIA FIRST PRIORITY FOR SHOPPING:
   - For purchase, availability, price, or seller questions ("Where can I buy this?", "Which seller is cheaper?"):
     Unless the user explicitly specified another country, ALWAYS prioritize Indian sellers and INR pricing (₹) (Amazon India, Flipkart, Official Indian Brand Stores, Decathlon India, Myntra, Croma, Reliance Digital, etc.).
   - If information is obtained from non-Indian sellers, explicitly label the country (e.g. "US Import").

3. DIRECT CLICKABLE PURCHASE & REFERENCE LINKS:
   - For purchase/shopping questions, provide direct clickable Markdown links: `[Seller Name / Link Text](URL)`.
   - NEVER INVENT FAKE DIRECT URLs (e.g. do not invent `amazon.in/dp/12345`). Use actual URLs found in web search evidence, or verified retailer search URLs preserving user item attributes:
     * Amazon India: `https://www.amazon.in/s?k={search_keywords}`
     * Flipkart: `https://www.flipkart.com/search?q={search_keywords}`
     * Google Shopping India: `https://www.google.co.in/search?tbm=shop&q={search_keywords}`
     * Decathlon India: `https://www.decathlon.in/search?query={search_keywords}`

4. STRUCTURED PRODUCT RESEARCH FORMAT (FOR SHOPPING/PURCHASE):
   When answering purchase/price questions, structure the answer clearly:
   
   ## Product Identified
   - **Product:** [Item / Model Name]
   - **Brand:** [Brand]
   - **Variant / Specs:** [Color, Storage, Size, etc.]
   - **Confidence:** [High / Medium / Low]

   ## Where to Buy in India
   | Seller | Price (INR) | Availability | Link |
   | :--- | :---: | :--- | :--- |
   | Amazon India | ₹... | In Stock | [View on Amazon India](https://...) |
   | Flipkart | ₹... | Available | [View on Flipkart](https://...) |
   | Official Store | ₹... | Available | [View Official Store](https://...) |

   ## Recommendation
   [Clear summary of which seller or option appears best and why]

   ## Important Notes
   [Mention warranty, variant differences, delivery limitations, or minor uncertainties]

5. MULTI-TURN CONVERSATION GROUNDING:
   - Maintain continuity across previous user turns. Recognize references like "its price", "where can I buy it", or "compare them" as referring to the item previously discussed.

6. ZERO HALLUCINATION & FACTUAL ACCURACY:
   - Do not invent non-existent models, prices, store listings, or URLs. Clearly distinguish confirmed facts from estimates.
"""

def execute_agent_research(request: AgentResearchRequest, api_key: str, pil_image: Optional[Image.Image] = None) -> AgentResearchResponse:
    """
    Executes the Gemini Ask AI Agent workflow:
    1. Resolves image data (from pil_image or request.image_base64)
    2. Runs Gemini Intent Detection & Query Planner (determining requires_image, requires_research, requires_high_quality_image, resolved_entity, search_query)
    3. Prepares high-quality image part if needed (using prepare_ask_ai_image_part)
    4. Executes web search tools if external research is needed (prioritizing India-first for shopping)
    5. Synthesizes detailed Gemini response with adaptive depth, markdown formatting, and clickable links.
    """
    if not api_key:
        raise ValueError("Gemini API key is missing.")

    client = genai.Client(api_key=api_key)
    used_tools: List[str] = []
    collected_sources: List[SourceItem] = []
    extracted_facts: List[FactItem] = []

    question = request.question.strip()
    image_ctx = request.image_context or {}
    conv_hist = request.conversation_history or []

    # Try resolving image from request base64 if pil_image not explicitly passed
    if pil_image is None and request.image_base64:
        pil_image = decode_base64_to_pil(request.image_base64)

    logger.info(f"[AskAIAgent] Processing question: '{question}' (Has image: {pil_image is not None})")

    # Step 1: Gemini Intent Detection & Query Planning
    intent_config = types.GenerateContentConfig(
        system_instruction=INTENT_DETECTION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=INTENT_RESPONSE_SCHEMA,
        temperature=0.0,
        max_output_tokens=1024,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    context_prompt = f"""User Question: "{question}"
Visual Analysis Context: {json.dumps(image_ctx, indent=2)}
Previous Conversation History: {json.dumps(conv_hist, indent=2)}

Determine the intent, whether image context is required, whether web research is required, whether it is a shopping query, resolve entity references, and build the targeted search query.
"""

    intent_str = "GENERAL_KNOWLEDGE"
    requires_research = False
    requires_image = (pil_image is not None)
    requires_hq_image = False
    is_shopping_query = False
    user_country = request.user_region
    search_query = ""
    resolved_entity_data = None

    try:
        intent_res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[context_prompt],
            config=intent_config
        )

        if intent_res and intent_res.text:
            cleaned = clean_json_text(intent_res.text)
            intent_data = json.loads(cleaned)
            intent_str = intent_data.get("intent", "GENERAL_KNOWLEDGE").upper()
            requires_research = bool(intent_data.get("requires_research", False))
            requires_image = bool(intent_data.get("requires_image", pil_image is not None))
            requires_hq_image = bool(intent_data.get("requires_high_quality_image", False))
            is_shopping_query = bool(intent_data.get("is_shopping_query", False))
            user_country = intent_data.get("user_country") or user_country
            search_query = intent_data.get("search_query", "").strip()
            resolved_entity_data = intent_data.get("resolved_entity")

            logger.info(
                f"[AskAIAgent] Intent: {intent_str} | requires_research: {requires_research} | "
                f"requires_image: {requires_image} | is_shopping: {is_shopping_query} | search_query: '{search_query}'"
            )
    except Exception as e:
        logger.warning(f"[AskAIAgent] Intent router fallback: {e}")
        # Intelligent fallback heuristics
        q_lower = question.lower()
        if any(w in q_lower for w in ["buy", "price", "cost", "seller", "shop", "where to"]):
            intent_str = "PRODUCT_PURCHASE"
            requires_research = True
            is_shopping_query = True
        elif any(w in q_lower for w in ["what is this", "color", "object", "show", "image"]):
            intent_str = "IMAGE_QUESTION"
            requires_image = (pil_image is not None)
            requires_research = False
        else:
            intent_str = "GENERAL_KNOWLEDGE"

    entity_obj = None
    if resolved_entity_data and isinstance(resolved_entity_data, dict):
        entity_obj = ProductEntity(
            name=resolved_entity_data.get("name"),
            brand=resolved_entity_data.get("brand"),
            model=resolved_entity_data.get("model"),
            variant=resolved_entity_data.get("variant"),
            category=resolved_entity_data.get("category"),
            confidence=0.95
        )

    # Prepare Image Part if image is required and available
    image_part = None
    if pil_image and (requires_image or intent_str in ["IMAGE_QUESTION", "PRODUCT_IDENTIFICATION"]):
        try:
            # Use high-quality preparation (up to 2048px/900KB) for fine details
            img_bytes, mime_type, w, h = prepare_ask_ai_image_part(pil_image)
            image_part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            used_tools.append("image_understanding")
            logger.info(f"[AskAIAgent] Encoded High-Quality Image Part ({w}x{h}, {len(img_bytes)/1024:.1f} KB)")
        except Exception as img_err:
            logger.warning(f"[AskAIAgent] Image encoding failed: {img_err}")

    # Step 2: Web Research Execution if required
    if requires_research and search_query:
        used_tools.append("web_search")
        web_tool = WebSearchTool()

        # India-First Priority for Shopping:
        # If user did NOT explicitly specify another country, prioritize India sellers
        has_explicit_country = bool(user_country or any(c in question.lower() for c in ["usa", "uk", "us", "canada", "germany", "japan", "australia", "singapore", "dubai", "uae"]))
        
        if (is_shopping_query or intent_str in ["PRODUCT_PURCHASE", "PRODUCT_PRICE", "PRODUCT_AVAILABILITY"]) and not has_explicit_country:
            if "india" not in search_query.lower():
                search_query = f"{search_query} India"
            logger.info(f"[AskAIAgent] Enforced INDIA FIRST search query: '{search_query}'")

        search_results = web_tool.execute(search_query, max_results=6)
        seen_urls = set()
        raw_sources_for_prompt = []

        for item in search_results:
            url = item.get("url", "").strip()
            if not url or not (url.startswith("http://") or url.startswith("https://")) or url in seen_urls:
                continue
            seen_urls.add(url)

            domain = item.get("domain") or extract_domain(url)
            stype = item.get("source_type") or classify_source_type(url, item.get("title", ""), item.get("snippet", ""))

            src_obj = SourceItem(
                title=item.get("title", "Web Source"),
                url=url,
                domain=domain,
                source_type=stype,
                snippet=item.get("snippet", "")
            )
            collected_sources.append(src_obj)
            raw_sources_for_prompt.append({
                "title": src_obj.title,
                "url": src_obj.url,
                "domain": src_obj.domain,
                "source_type": src_obj.source_type,
                "snippet": src_obj.snippet
            })

        # For shopping queries, construct working retailer search links for top Indian marketplaces if missing
        if is_shopping_query or intent_str in ["PRODUCT_PURCHASE", "PRODUCT_PRICE", "PRODUCT_AVAILABILITY"]:
            item_keyword = (
                entity_obj.name if (entity_obj and entity_obj.name) 
                else search_query.replace("India", "").strip() or question
            )
            retailers = [
                ("Amazon India", "amazon.in"),
                ("Flipkart", "flipkart.com"),
                ("Croma", "croma.com"),
                ("Reliance Digital", "reliancedigital.in")
            ]
            for ret_name, dom in retailers:
                if not any(dom in s.domain for s in collected_sources):
                    ret_url = build_retailer_search_link(ret_name, item_keyword)
                    src_obj = SourceItem(
                        title=f"{ret_name} Search: {item_keyword}",
                        url=ret_url,
                        domain=dom,
                        source_type="retailer",
                        snippet=f"Search listings for '{item_keyword}' on {ret_name}"
                    )
                    collected_sources.append(src_obj)
                    raw_sources_for_prompt.append({
                        "title": src_obj.title,
                        "url": src_obj.url,
                        "domain": src_obj.domain,
                        "source_type": src_obj.source_type,
                        "snippet": src_obj.snippet
                    })

        # Optional Page Reader Tool for top candidate if price/purchase/spec details requested
        page_text = None
        if collected_sources and any(k in question.lower() for k in ["price", "buy", "purchase", "spec", "compare", "cost"]):
            top_url = collected_sources[0].url
            try:
                reader = PageReaderTool()
                page_text = reader.execute(top_url, max_chars=1800)
                if page_text:
                    used_tools.append("page_reader")
            except Exception as pr_err:
                logger.warning(f"[AskAIAgent] Page reader skipped: {pr_err}")

    # Step 3: Synthesis with Gemini (Adaptive Depth + Grounded Evidence)
    synthesis_config = types.GenerateContentConfig(
        system_instruction=RESEARCH_SYNTHESIS_SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=4096,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    contents_payload = []
    if image_part:
        contents_payload.append(image_part)

    prompt_details = f"""User Question: "{question}"
Intent Category: {intent_str}
Is Shopping Query: {is_shopping_query}
User Country Override: {user_country or 'None (Default: India)'}
Resolved Entity: {entity_obj.model_dump_json() if entity_obj else 'None'}

Web Search Evidence ({len(collected_sources)} sources):
{json.dumps([s.model_dump() for s in collected_sources], indent=2)}

Visual Analysis Context:
{json.dumps(image_ctx, indent=2)}

Previous Conversation History:
{json.dumps(conv_hist, indent=2)}

Provide a complete, structured, and detailed answer appropriate for the question complexity.
"""
    contents_payload.append(prompt_details)

    try:
        synth_res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents_payload,
            config=synthesis_config
        )

        final_answer = synth_res.text.strip() if (synth_res and synth_res.text) else "Unable to generate response."
        res_summary = (
            f"Researched '{search_query}' across {len(collected_sources)} sources." 
            if requires_research else "Answered using visual reasoning and general knowledge."
        )
        confidence_val = "high" if (not requires_research or len(collected_sources) >= 2) else "medium"

        return AgentResearchResponse(
            success=True,
            answer=final_answer,
            intent=intent_str,
            requires_research=requires_research,
            entity=entity_obj,
            facts=extracted_facts,
            used_tools=used_tools,
            sources=collected_sources,
            confidence=confidence_val,
            research_summary=res_summary
        )
    except Exception as synth_err:
        logger.error(f"[AskAIAgent] Response synthesis error: {synth_err}", exc_info=True)
        return AgentResearchResponse(
            success=False,
            answer=f"I encountered an issue processing your request: {str(synth_err)}. Please try asking again.",
            intent=intent_str,
            requires_research=requires_research,
            entity=entity_obj,
            facts=[],
            used_tools=used_tools,
            sources=collected_sources,
            confidence="low",
            error=str(synth_err)
        )

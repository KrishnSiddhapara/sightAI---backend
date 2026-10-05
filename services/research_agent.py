import json
import logging
import time
from typing import List, Dict, Any, Optional
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

logger = logging.getLogger(__name__)

import urllib.parse

def build_retailer_search_link(retailer_name: str, product_query: str) -> str:
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
    else:
        return f"https://www.google.co.in/search?tbm=shop&q={q}"

INTENT_DETECTION_SYSTEM_PROMPT = """You are SightAI's Advanced Intent Classification & Query Planning Engine.
Your task is to examine the user's question, previous conversation history, and available image context, and classify the user's intent:

INTENT CATEGORIES:
1. PURCHASE_RESEARCH: Questions asking where to buy, purchase options, deals, shopping links, or seller comparisons. Set requires_research=true.
2. PRODUCT_PRICE: Current pricing info, deals, or market rates. Set requires_research=true.
3. PRODUCT_AVAILABILITY: Retailers, stock, or purchasing options. Set requires_research=true.
4. PRODUCT_SPECIFICATION: Technical specs, processor, RAM, materials, dimensions. Set requires_research=true.
5. PRODUCT_COMPARISON: Comparing items/products (e.g. "Compare this with iPhone 17 Pro"). Set requires_research=true.
6. LATEST_INFORMATION: Latest release, current news, updates, announcements, 2026 developments. Set requires_research=true.
7. COMPANY_RESEARCH: Manufacturer / publisher official company info. Set requires_research=true.
8. PERSON_RESEARCH: Public figures, authors, creators. Set requires_research=true if external info needed.
9. TECHNICAL_RESEARCH: Frameworks, libraries, documentation, APIs. Set requires_research=true.
10. IMAGE_ANALYSIS: Visual-only questions about the image (e.g. shirt color, counting objects, visible text). Set requires_research=false.
11. GENERAL_KNOWLEDGE: Static definitions or general concepts (e.g. "What is an API?", "What is RAG?"). Set requires_research=false.
12. OTHER: Any other request.

CRITICAL RULES FOR SEARCH QUERY GENERATION:
- INDIA FIRST PRIORITY: Unless the user explicitly specifies another target country (e.g., "in USA", "in UK"), construct targeted search queries prioritizing INDIA sellers, pricing in INR (₹), and Indian marketplaces (e.g. "Amazon India Flipkart").
- PRESERVE USER KEYWORDS: Retain exact item keywords (brand, color, item type, model, size, variant, storage) provided by the user or identified in visual context. (e.g., "Nike red football buy online India", "Samsung S25 Ultra 512GB Black price India").
- PRONOUN RESOLUTION: Resolve pronouns like "its price", "that phone", "where to buy it" using previous conversation turns and detected visual entities.
"""

INTENT_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "intent": {
            "type": "STRING",
            "description": "PURCHASE_RESEARCH, PRODUCT_PRICE, PRODUCT_AVAILABILITY, PRODUCT_SPECIFICATION, PRODUCT_COMPARISON, LATEST_INFORMATION, COMPANY_RESEARCH, PERSON_RESEARCH, TECHNICAL_RESEARCH, IMAGE_ANALYSIS, GENERAL_KNOWLEDGE, or OTHER"
        },
        "requires_research": {"type": "BOOLEAN"},
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
        "search_query": {"type": "STRING", "description": "Targeted search query for web_search tool prioritizing India sellers."},
        "reasoning": {"type": "STRING"}
    },
    "required": ["intent", "requires_research", "search_query"]
}

RESEARCH_SYNTHESIS_SYSTEM_PROMPT = """You are SightAI's Advanced General AI Assistant & Enterprise Research Agent, inspired by top AI assistants (ChatGPT/Gemini).

CORE BEHAVIOR & SHOPPING / RESEARCH POLICY:

1. INDIA FIRST PRIORITY FOR SHOPPING:
   - For purchase, availability, price, or seller questions ("Where can I buy this?", "Where to buy this ball?", "Price of this phone"):
     Always prioritize INDIAN sellers and pricing in INR (₹) first (Amazon India, Flipkart, Official Indian Brand Stores, Decathlon India, Myntra, Croma, Reliance Digital, etc.) unless the user explicitly asks for another country.
   - International sellers should only be listed if Indian options are genuinely unavailable or explicitly requested, with clear country labels (e.g., "US Import").

2. DIRECT CLICKABLE PURCHASE LINKS & RETAILER SEARCH URLs:
   - For purchase/shopping questions, ALWAYS provide direct, clickable Markdown links to purchase options: `[Seller Name / Link Text](URL)`.
   - NO FAKE DIRECT URLs: NEVER invent unverified product page URLs (e.g. `amazon.in/product/12345`).
   - Use verified direct product URLs obtained from search evidence.
   - If exact direct product URL is not verified, use retailer search URLs:
     * Amazon India: `https://www.amazon.in/s?k={search_keywords}`
     * Flipkart: `https://www.flipkart.com/search?q={search_keywords}`
     * Google Shopping India: `https://www.google.co.in/search?tbm=shop&q={search_keywords}`
     * Decathlon India: `https://www.decathlon.in/search?query={search_keywords}`

3. STRUCTURED PRODUCT RESULT FORMAT (FOR SHOPPING):
   Use a clean, readable layout:
   ### Product Identified
   **Product:** [Identified Item/Model Name]
   **Brand:** [Brand] | **Variant:** [Color / Size / Spec]

   ### Where to Buy in India
   | Retailer / Seller | Verified / Listed Price (INR) | Availability | Purchase Link |
   | :--- | :---: | :--- | :--- |
   | Amazon India | ₹1,499 | In Stock | [View on Amazon India](https://www.amazon.in/s?k=...) |
   | Flipkart | ₹1,599 | In Stock | [View on Flipkart](https://www.flipkart.com/search?q=...) |
   | Official Store | ₹1,695 | Available | [Official Store](...) |

4. ADAPTIVE ANSWER DEPTH:
   - Match answer depth to question complexity.
   - Simple trivial questions (e.g. "What is 2+2?", "What color is this shirt?") get a direct, simple response.
   - Technical, conceptual, programming, AI/ML, product research, or comparison questions receive a complete, detailed, well-structured markdown explanation with headers, bullet points, tables, and code snippets where appropriate.

5. ZERO HALLUCINATION & FACTUAL ACCURACY:
   - Never invent facts, prices, specifications, URLs, or store names. Distinguish verified facts from uncertainty clearly.
"""

def execute_agent_research(request: AgentResearchRequest, api_key: str) -> AgentResearchResponse:
    if not api_key:
        raise ValueError("API key is missing.")

    client = genai.Client(api_key=api_key)
    used_tools: List[str] = []
    collected_sources: List[SourceItem] = []
    extracted_facts: List[FactItem] = []

    question = request.question.strip()
    image_ctx = request.image_context or {}
    conv_hist = request.conversation_history or []

    logger.info(f"[ResearchAgent] Processing query: '{question}'")

    # Step 1: Intent & Entity Router
    intent_config = types.GenerateContentConfig(
        system_instruction=INTENT_DETECTION_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=INTENT_RESPONSE_SCHEMA,
        temperature=0.0,
        max_output_tokens=1024,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    context_prompt = f"""User Question: "{question}"
Image Analysis Context: {json.dumps(image_ctx, indent=2)}
Previous Conversation Turns: {json.dumps(conv_hist, indent=2)}

Determine the intent, resolve entity references, and generate a search query if external research is required.
"""

    intent_str = "GENERAL_KNOWLEDGE"
    requires_research = False
    search_query = ""
    resolved_entity_data = None

    try:
        res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[context_prompt],
            config=intent_config
        )

        if res and res.text:
            cleaned = clean_json_text(res.text)
            intent_data = json.loads(cleaned)
            intent_str = intent_data.get("intent", "GENERAL_KNOWLEDGE").upper()
            requires_research = bool(intent_data.get("requires_research", False))
            search_query = intent_data.get("search_query", "").strip()
            resolved_entity_data = intent_data.get("resolved_entity")
            logger.info(f"[ResearchAgent] Intent: {intent_str}, requires_research={requires_research}, search_query='{search_query}'")
    except Exception as e:
        logger.warning(f"[ResearchAgent] Intent router glitch: {e}. Defaulting to visual context execution.")
        requires_research = False

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

    # Case A: Visual Question or General Knowledge -> Direct LLM Answer with adaptive depth
    if not requires_research or not search_query:
        direct_config = types.GenerateContentConfig(
            system_instruction=RESEARCH_SYNTHESIS_SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=4096,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        visual_prompt = f"""User Question: "{question}"
Intent Category: {intent_str}
Resolved Entity: {entity_obj.model_dump_json() if entity_obj else 'None'}
Image Analysis Context: {json.dumps(image_ctx, indent=2)}
Previous Conversation History: {json.dumps(conv_hist, indent=2)}

Provide a complete, structured, and detailed response appropriate for the question complexity. Use headings, markdown formatting, bullet points, and code examples when helpful.
"""
        ans_res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[visual_prompt],
            config=direct_config,
        )
        final_ans = ans_res.text.strip() if ans_res and ans_res.text else "Information not available."
        
        return AgentResearchResponse(
            success=True,
            answer=final_ans,
            intent=intent_str,
            requires_research=False,
            entity=entity_obj,
            facts=[],
            used_tools=[],
            sources=[],
            confidence="high",
            research_summary="Answered directly using intelligent general knowledge and visual context."
        )

    # Case B: External Research Required -> Execute Web Search Tool
    used_tools.append("web_search")
    web_tool = WebSearchTool()

    # Ensure India-First priority for shopping and purchase requests
    is_shopping_query = any(kw in question.lower() or kw in search_query.lower() for kw in ["buy", "purchase", "price", "cost", "available", "seller", "store", "deal", "where to", "where can i"])
    has_explicit_country = any(c in question.lower() or c in search_query.lower() for c in ["usa", "uk", "us", "canada", "germany", "japan", "australia", "singapore", "dubai", "uae"])

    if is_shopping_query and not has_explicit_country and "india" not in search_query.lower():
        search_query = f"{search_query} India"
        logger.info(f"[ResearchAgent] Applied INDIA FIRST query priority: '{search_query}'")

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

    # For shopping queries, append working retailer search URLs for Indian marketplaces if direct retailer links are missing
    if is_shopping_query:
        product_keyword = (entity_obj.name if entity_obj and entity_obj.name else search_query.replace("India", "").strip())
        retailers = [("Amazon India", "amazon.in"), ("Flipkart", "flipkart.com")]
        for ret_name, dom in retailers:
            if not any(dom in s.domain for s in collected_sources):
                ret_url = build_retailer_search_link(ret_name, product_keyword)
                src_obj = SourceItem(
                    title=f"{ret_name} Search: {product_keyword}",
                    url=ret_url,
                    domain=dom,
                    source_type="retailer",
                    snippet=f"Search listings for '{product_keyword}' on {ret_name}"
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
            logger.warning(f"[ResearchAgent] Page reader skipped: {pr_err}")

    # Step 3: Synthesis & Fact Extraction
    synthesis_config = types.GenerateContentConfig(
        system_instruction=RESEARCH_SYNTHESIS_SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=4096,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    synthesis_prompt = f"""User Question: "{question}"
Intent Category: {intent_str}
Resolved Entity: {entity_obj.model_dump_json() if entity_obj else 'None'}

Web Search Evidence ({len(collected_sources)} sources):
{json.dumps(raw_sources_for_prompt, indent=2)}

Page Reader Content (Top Source):
{page_text or 'Not executed'}

Image Context:
{json.dumps(image_ctx, indent=2)}

Provide a clear, grounded answer to the user's question using the verified search evidence. Include clickable Markdown links to verified sources where relevant.
"""

    try:
        synth_res = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[synthesis_prompt],
            config=synthesis_config
        )

        final_answer = synth_res.text.strip() if synth_res and synth_res.text else "Unable to synthesize answer from external sources."
        
        res_summary = f"Researched '{search_query}' across {len(collected_sources)} verified external sources."
        confidence_val = "high" if len(collected_sources) >= 2 else ("medium" if len(collected_sources) == 1 else "low")

        return AgentResearchResponse(
            success=True,
            answer=final_answer,
            intent=intent_str,
            requires_research=True,
            entity=entity_obj,
            facts=extracted_facts,
            used_tools=used_tools,
            sources=collected_sources,
            confidence=confidence_val,
            research_summary=res_summary
        )
    except Exception as synth_err:
        logger.error(f"[ResearchAgent] Synthesis error: {synth_err}")
        return AgentResearchResponse(
            success=False,
            answer="I encountered an issue while processing external research results. Please try again.",
            intent=intent_str,
            requires_research=True,
            entity=entity_obj,
            facts=[],
            used_tools=used_tools,
            sources=collected_sources,
            confidence="low",
            error=str(synth_err)
        )

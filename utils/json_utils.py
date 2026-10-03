import re

def clean_json_text(text: str) -> str:
    """
    Cleans raw LLM response strings by stripping markdown code block wrappers
    (e.g. ```json ... ``` or ``` ... ```) and trimming whitespace so json.loads
    can reliably parse it without throwing JSONDecodeError.
    """
    if not text:
        return ""
    cleaned = text.strip()
    
    # Strip markdown fences if present
    pattern = r"^```(?:json)?\s*([\s\S]*?)\s*```$"
    match = re.match(pattern, cleaned, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
        
    return cleaned

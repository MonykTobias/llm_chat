import html  # add to the imports at the top

def _is_ollama_xml_bug(e: Exception) -> bool:
    s = str(e)
    return "XML syntax error" in s or "tool call parsing failed" in s

def _safe_invoke(llm, msgs, tries: int = 2):
    """Re-roll on Ollama's server-side tool-call XML parse failure."""
    for n in range(tries):
        try:
            return llm.invoke(msgs)
        except Exception as e:
            if _is_ollama_xml_bug(e) and n < tries - 1:
                continue
            raise

def _scrub(text: str) -> str:
    """Escape angle brackets so file contents don't feed raw <,> back into the next tool call."""
    return html.escape(text or "", quote=False)
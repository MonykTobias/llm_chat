"""
Central LLM factory.

Every agent talks to a local Ollama server. We use ChatOllama (the native
/api/chat client) instead of ChatOpenAI against Ollama's /v1 OpenAI-compat shim
because only the native API honors num_ctx. Through the /v1 endpoint the nested
`options` object (where num_ctx lives) is dropped, so every model silently loads
at Ollama's 4096-token default and long prompts are truncated or rejected
("exceeds the available context size (4096 tokens)").

ChatOllama maps our per-agent config.yaml values straight onto the request:
  context_window -> num_ctx        (the actual loaded context size)
  max_tokens     -> num_predict    (generation cap)
  thinking       -> reasoning      (enable model thinking; only sent when true)
so each agent's configured context finally takes effect. Verify with `ollama ps`
— the CONTEXT column should now match config.yaml.
"""
import re

from langchain_core.callbacks import BaseCallbackHandler
from langchain_ollama import ChatOllama

def _strip_thinking(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

class _StripThinkingCallback(BaseCallbackHandler):
    """Strip <think>...</think> from LLM output before LangChain parses tool calls."""

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for gen in generations:
                if hasattr(gen, "text"):
                    gen.text = _strip_thinking(gen.text)
                if hasattr(gen, "message") and isinstance(gen.message.content, str):
                    gen.message.content = _strip_thinking(gen.message.content)

def _native_base_url(base_url: str) -> str:
    """Strip the OpenAI-compat '/v1' suffix — ChatOllama uses the native API root."""
    url = (base_url or "").rstrip("/")
    if url.endswith("/v1"):
        url = url[: -len("/v1")].rstrip("/")
    return url or "http://localhost:11434"

def make_llm(cfg: dict) -> ChatOllama:
    """Build a ChatOllama from one `agents.<name>` section of config.yaml."""
    kwargs = dict(
        model=cfg["model"],
        base_url=_native_base_url(cfg["base_url"]),
        temperature=cfg["temperature"],
        num_predict=cfg["max_tokens"],
        num_ctx=cfg["context_window"],
        num_gpu=-1,
    )
    # Only send `reasoning` for models that support thinking; passing it to a
    # non-thinking model would push an unsupported `think` flag to Ollama.
    if thinking := cfg.get("thinking"):
        if cfg.get("thinking_levels"):
            # GPT-OSS style: low/medium/high
            kwargs["reasoning"] = thinking if isinstance(thinking, str) else "medium"
        else:
            # Qwen / binary models: just on/off
            kwargs["reasoning"] = True
        # callbacks are handled by langchain (start/end of llm/tools)
        kwargs["callbacks"] = [_StripThinkingCallback()]
    elif thinking is not None:
        kwargs["reasoning"] = False

    return ChatOllama(**kwargs)


def make_system_prompt(base_prompt: str, cfg: dict) -> str:
    """Prepend <|think|> for models that use prompt-based thinking (e.g. Gemma 4)."""
    if cfg.get("thinking_token"):
        return "<|think|>\n" + base_prompt
    return base_prompt
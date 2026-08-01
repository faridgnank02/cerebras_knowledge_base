"""LLM provider selection.

The distiller, reranker, planner, and synthesizer all talk to an LLM through the
OpenAI Python SDK's ``chat.completions.create`` surface. That works directly for
any OpenAI-compatible endpoint (Cerebras, OpenAI itself, or a local server).

To support Claude we do *not* point the OpenAI SDK at Anthropic — Claude is
called through the official ``anthropic`` SDK. ``AnthropicChatAdapter`` wraps that
SDK behind the same minimal chat surface the components consume, so provider
choice stays a config switch and the call sites don't change.

Provider is set in ``config.yaml`` under ``llm.provider``:

    llm:
      provider: openai      # Cerebras / OpenAI / any OpenAI-compatible base_url
      # provider: anthropic # Claude via the official anthropic SDK
"""

import os

# Env var checked per provider, in order; first non-empty wins. LLM_API_KEY is
# the provider-neutral override; the rest are conventional per-provider names
# (CEREBRAS_API_KEY kept for backward compatibility).
_KEY_ENV = {
    "anthropic": ("LLM_API_KEY", "ANTHROPIC_API_KEY"),
    "openai": ("LLM_API_KEY", "OPENAI_API_KEY", "CEREBRAS_API_KEY"),
}


class LLMKeyError(RuntimeError):
    """No API key found in the environment for the configured provider."""

    def __init__(self, provider: str):
        self.provider = provider
        names = ", ".join(_key_env_names(provider))
        super().__init__(f"No LLM API key set for provider '{provider}' (looked for: {names})")


def _normalize(provider: str) -> str:
    # "cerebras" is just the OpenAI-compatible path with a Cerebras base_url.
    p = (provider or "openai").lower()
    return "openai" if p in ("openai", "cerebras", "openai_compatible") else p


def _key_env_names(provider: str) -> tuple[str, ...]:
    return _KEY_ENV.get(_normalize(provider), _KEY_ENV["openai"])


def resolve_api_key(provider: str) -> str | None:
    """Return the API key for ``provider`` from the environment, or None."""
    for name in _key_env_names(provider):
        value = os.environ.get(name)
        if value:
            return value
    return None


class _Msg:
    def __init__(self, content: str):
        self.content = content


class _Choice:
    def __init__(self, content: str):
        self.message = _Msg(content)


class _Resp:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class AnthropicChatAdapter:
    """Presents the ``client.chat.completions.create(...)`` surface the LLM
    components use, but calls Claude through the official ``anthropic`` SDK.

    Translates OpenAI-style ``response_format`` (strict json_schema) into
    Anthropic structured outputs (``output_config.format``), and returns a
    response object shaped like the OpenAI one (``.choices[0].message.content``).
    """

    def __init__(self, api_key: str, max_tokens: int = 4096, client=None):
        if client is None:
            # lazy so the anthropic dep is only needed when the provider is used
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
        self._client = client
        self.max_tokens = max_tokens
        # so component code's `client.chat.completions.create(...)` resolves here
        self.chat = self
        self.completions = self

    def create(self, *, model, messages, response_format=None, **_ignored):
        system = "\n\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        convo = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] != "system"
        ]
        kwargs = {"model": model, "max_tokens": self.max_tokens, "messages": convo}
        if system:
            kwargs["system"] = system
        if response_format is not None:
            schema = response_format["json_schema"]["schema"]
            kwargs["output_config"] = {
                "format": {"type": "json_schema", "schema": schema}
            }
        resp = self._client.messages.create(**kwargs)
        text = next(
            (b.text for b in resp.content if getattr(b, "type", None) == "text"),
            "",
        )
        return _Resp(text)


def build_llm_client(cfg, *, api_key: str | None = None):
    """Build the LLM client for the configured provider.

    Raises ``LLMKeyError`` if no API key is available.
    """
    provider = _normalize(getattr(cfg, "llm_provider", "openai"))
    key = api_key or resolve_api_key(provider)
    if not key:
        raise LLMKeyError(getattr(cfg, "llm_provider", "openai"))
    if provider == "anthropic":
        return AnthropicChatAdapter(key, max_tokens=getattr(cfg, "llm_max_tokens", 4096))
    from openai import OpenAI

    return OpenAI(base_url=cfg.llm_base_url, api_key=key)

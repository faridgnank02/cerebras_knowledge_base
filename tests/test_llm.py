import json

import pytest

from knowbase.llm import (
    AnthropicChatAdapter,
    LLMKeyError,
    build_llm_client,
    resolve_api_key,
)

ALL_KEY_ENVS = [
    "LLM_API_KEY", "OPENAI_API_KEY", "CEREBRAS_API_KEY", "ANTHROPIC_API_KEY",
]


@pytest.fixture
def clean_env(monkeypatch):
    for name in ALL_KEY_ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


class Cfg:
    def __init__(self, provider="openai", model="m", base_url="http://x", max_tokens=4096):
        self.llm_provider = provider
        self.llm_model = model
        self.llm_base_url = base_url
        self.llm_max_tokens = max_tokens


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic with a .messages.create surface."""

    def __init__(self, text="RESULT"):
        self.text = text
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("R", (), {"content": [_Block(self.text)]})()


# --- resolve_api_key ---------------------------------------------------------

def test_openai_key_precedence_prefers_llm_api_key(clean_env):
    clean_env.setenv("CEREBRAS_API_KEY", "cbr")
    clean_env.setenv("LLM_API_KEY", "neutral")
    assert resolve_api_key("openai") == "neutral"


def test_openai_falls_back_to_cerebras_key(clean_env):
    clean_env.setenv("CEREBRAS_API_KEY", "cbr")
    assert resolve_api_key("openai") == "cbr"


def test_cerebras_provider_is_openai_compatible(clean_env):
    clean_env.setenv("CEREBRAS_API_KEY", "cbr")
    assert resolve_api_key("cerebras") == "cbr"


def test_anthropic_uses_anthropic_key_not_cerebras(clean_env):
    clean_env.setenv("CEREBRAS_API_KEY", "cbr")
    clean_env.setenv("ANTHROPIC_API_KEY", "ant")
    assert resolve_api_key("anthropic") == "ant"


def test_no_key_returns_none(clean_env):
    assert resolve_api_key("openai") is None
    assert resolve_api_key("anthropic") is None


# --- build_llm_client --------------------------------------------------------

def test_build_raises_when_no_key(clean_env):
    with pytest.raises(LLMKeyError):
        build_llm_client(Cfg(provider="anthropic"))


def test_build_openai_client_for_openai_provider(clean_env):
    client = build_llm_client(Cfg(provider="openai"), api_key="k")
    assert not isinstance(client, AnthropicChatAdapter)
    assert hasattr(client.chat.completions, "create")


def test_build_anthropic_adapter_for_anthropic_provider(clean_env):
    client = build_llm_client(Cfg(provider="anthropic", max_tokens=999), api_key="k")
    assert isinstance(client, AnthropicChatAdapter)
    assert client.max_tokens == 999


# --- AnthropicChatAdapter translation ----------------------------------------

def _adapter(text="RESULT", max_tokens=1024):
    fake = FakeAnthropic(text)
    return AnthropicChatAdapter("k", max_tokens=max_tokens, client=fake), fake


def test_adapter_translates_system_and_user_and_json_schema():
    adapter, fake = _adapter()
    resp = adapter.chat.completions.create(
        model="claude-x",
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U"},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "n", "strict": True, "schema": {"type": "object"}},
        },
    )
    assert resp.choices[0].message.content == "RESULT"
    call = fake.calls[0]
    assert call["model"] == "claude-x"
    assert call["system"] == "SYS"
    assert call["messages"] == [{"role": "user", "content": "U"}]
    assert call["max_tokens"] == 1024
    assert call["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}}
    }


def test_adapter_omits_output_config_for_plain_text():
    adapter, fake = _adapter()
    adapter.chat.completions.create(
        model="claude-x",
        messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "U"},
        ],
    )
    assert "output_config" not in fake.calls[0]


def test_distiller_works_through_anthropic_adapter():
    from knowbase.ingest.distill import Distiller

    good = json.dumps(
        {"question": "q", "summary": "s", "resolution": "r", "systems": [], "code_refs": []}
    )
    adapter, fake = _adapter(text=good)
    d = Distiller("", "", "claude-opus-5", client=adapter, sleep=lambda s: None)
    art = d.distill("title", "thread body")
    assert art.question == "q"
    call = fake.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["output_config"]["format"]["type"] == "json_schema"

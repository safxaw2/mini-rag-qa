from types import SimpleNamespace

from src.llm import llm as llm_module
from src.llm.llm import LLM


def test_init_configures_openai_client(monkeypatch):
    init_call = {}

    class StubOpenAI:
        def __init__(self, **kwargs):
            init_call.update(kwargs)

    monkeypatch.setattr(llm_module, "OpenAI", StubOpenAI)

    llm = LLM("test-model", "test-key", "https://example.com/v1")

    assert llm.model == "test-model"
    assert llm.api_key == "test-key"
    assert llm.base_url == "https://example.com/v1"
    assert isinstance(llm.client, StubOpenAI)
    assert init_call == {
        "api_key": "test-key",
        "base_url": "https://example.com/v1",
    }


def test_generate_sends_request_and_formats_response(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="生成的回答"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            total_tokens=15,
            prompt_tokens=9,
            completion_tokens=6,
        ),
        model="test-model",
    )
    request = {}

    class StubCompletions:
        def create(self, **kwargs):
            request.update(kwargs)
            return response

    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=StubCompletions(),
        )
    )
    messages = [{"role": "user", "content": "测试问题"}]

    result = llm.generate(messages, temperature=0.3, max_tokens=128)

    assert request == {
        "model": "test-model",
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 128,
    }
    assert result == {
        "content": "生成的回答",
        "finish_reason": "stop",
        "tokens_used": {
            "total": 15,
            "prompt": 9,
            "completion": 6,
        },
        "model": "test-model",
    }


def test_generate_uses_default_options():
    request = {}
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="回答"),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(
            total_tokens=4,
            prompt_tokens=2,
            completion_tokens=2,
        ),
        model="test-model",
    )

    class StubCompletions:
        def create(self, **kwargs):
            request.update(kwargs)
            return response

    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=StubCompletions(),
        )
    )

    llm.generate([{"role": "user", "content": "问题"}])

    assert request["temperature"] == 0.5
    assert request["max_tokens"] == 2000


def test_generate_stream_yields_nonempty_content(monkeypatch):
    requests = []
    chunks = [
        SimpleNamespace(choices=[
            SimpleNamespace(delta=SimpleNamespace(content="流"))
        ]),
        SimpleNamespace(choices=[
            SimpleNamespace(delta=SimpleNamespace(content=None))
        ]),
        SimpleNamespace(choices=[
            SimpleNamespace(delta=SimpleNamespace(content="式答案"))
        ]),
    ]

    class StubCompletions:
        def create(self, **kwargs):
            requests.append(kwargs)
            return iter(chunks)

    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=StubCompletions())
    )

    assert list(llm.generate_stream([{"role": "user", "content": "问题"}])) == [
        "流",
        "式答案",
    ]
    assert requests[0]["stream"] is True
    assert requests[0]["model"] == "test-model"

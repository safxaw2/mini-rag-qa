from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.main import app
from src.api import deps
from src.api.config import settings
from src.api.routers import chat as chat_router


def test_get_llm_resolves_configured_model_name(monkeypatch):
    settings_provider = {
        "qwen": {
            "api_key": "test-key",
            "base_url": "https://example.com/v1",
            "default_model": "qwen3.7-flash",
        }
    }
    created = {}

    class StubLLM:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(settings, "LLM_PROVIDER", settings_provider)
    monkeypatch.setattr(deps, "LLM", StubLLM)

    llm = deps.get_llm("qwen3.7-flash")
    cached_llm = deps.get_llm("qwen3.7-flash")

    assert isinstance(llm, StubLLM)
    assert cached_llm is llm
    assert created == {
        "model": "qwen3.7-flash",
        "api_key": "test-key",
        "base_url": "https://example.com/v1",
    }
    deps._get_llm.cache_clear()


def test_embedding_factory_caches_local_and_online_models(monkeypatch):
    created = []

    class StubEmbedding:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(deps, "LocalEmbedding", StubEmbedding)
    monkeypatch.setattr(deps, "OpenAIEmbedding", StubEmbedding)
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")
    monkeypatch.setattr(settings, "LOCAL_EMBEDDING_MODEL", "local-test")
    monkeypatch.setattr(settings, "LOCAL_EMBEDDING_DEVICE", "cpu")
    deps._get_embedding_model.cache_clear()

    local_first = deps.get_embedding_model()
    local_second = deps.get_embedding_model()
    assert local_first is local_second
    assert created == [{"model_name": "local-test", "device": "cpu"}]

    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "online-test")
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "test-key")
    monkeypatch.setattr(settings, "EMBEDDING_BASE_URL", "https://example.com/v1")
    monkeypatch.setattr(settings, "EMBEDDING_DIMENSION", 512)

    online_first = deps.get_embedding_model()
    online_second = deps.get_embedding_model()
    assert online_first is online_second
    assert created[-1] == {
        "api_key": "test-key",
        "model": "online-test",
        "base_url": "https://example.com/v1",
        "dimension": 512,
    }
    deps._get_embedding_model.cache_clear()


def test_chat_endpoint_returns_sources_and_total_token_count(monkeypatch):
    class StubLLM:
        def generate(self, messages, temperature, max_tokens):
            return {
                "content": "这是答案",
                "tokens_used": {"total": 17},
                "model": "test-model",
            }

    content = "参考内容" * 60
    search_response = SimpleNamespace(
        results=[
            {"id": "doc-1", "score": 0.1, "content": content},
        ]
    )
    monkeypatch.setattr(chat_router, "get_llm", lambda model: StubLLM())
    monkeypatch.setattr(
        chat_router,
        "get_search_service",
        lambda collection_name: SimpleNamespace(
            search=lambda query, config: search_response
        ),
    )

    response = TestClient(app).post(
        "/api/chat/",
        json={"query": "问题", "collection_name": "documents"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"] == "问题"
    assert body["answer"] == "这是答案"
    assert body["tokens_used"] == 17
    assert body["sources"] == [
        {"id": "doc-1", "score": 0.1, "text": content[:200] + "..."}
    ]


def test_chat_endpoint_without_search_results_uses_zero_tokens(monkeypatch):
    class UnusedLLM:
        def generate(self, *args, **kwargs):
            raise AssertionError("LLM should not be called without search results")

    monkeypatch.setattr(chat_router, "get_llm", lambda model: UnusedLLM())
    monkeypatch.setattr(
        chat_router,
        "get_search_service",
        lambda collection_name: SimpleNamespace(
            search=lambda query, config: SimpleNamespace(results=[])
        ),
    )

    response = TestClient(app).post(
        "/api/chat/",
        json={"query": "问题", "collection_name": "documents"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "没有找到答案"
    assert response.json()["tokens_used"] == 0
    assert response.json()["sources"] == []


def test_stream_chat_endpoint_returns_sse_tokens_and_sources(monkeypatch):
    class StubLLM:
        def generate_stream(self, messages, temperature, max_tokens):
            assert messages[0]["role"] == "user"
            yield "流式"
            yield "回答"

    monkeypatch.setattr(chat_router, "get_llm", lambda model: StubLLM())
    monkeypatch.setattr(
        chat_router,
        "get_search_service",
        lambda collection_name: SimpleNamespace(
            search=lambda query, config: SimpleNamespace(
                results=[{
                    "id": "doc-1",
                    "score": 0.1,
                    "content": "参考资料",
                }]
            )
        ),
    )

    response = TestClient(app).post(
        "/api/chat/stream",
        json={"query": "问题", "collection_name": "documents"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "token", "content": "流式"' in response.text
    assert '"type": "token", "content": "回答"' in response.text
    assert '"type": "done"' in response.text
    assert '"id": "doc-1"' in response.text

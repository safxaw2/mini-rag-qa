"""SearchService 测试。

分两层：
  1. 单元测试：FakeDocumentStore 记录调用参数、返回预置结果，
     验证 SearchService 的默认配置、参数透传、结果结构、过滤、计时与异常
  2. 集成测试：真实 DocumentStore + ChromaVectorStore（配假嵌入），
     验证与真实依赖的契约（namespace / metadata_filter / 结果键）

注意：项目里的 score 是 Chroma 返回的余弦距离，越小越相似，
所以阈值过滤使用 max_distance，只保留 score <= max_distance 的结果。

运行方式：
    pytest tests/test_search_service.py -v -s
    # 或直接运行
    python tests/test_search_service.py
"""

import os
import sys
import time
import uuid

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK，统一切到 UTF-8，避免打印中文时乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from src.database.document_store import DocumentStore
from src.embeddings.base import BaseEmbedding
from src.embeddings.chroma_store import ChromaVectorStore
from src.retrieval.search_service import (
    RetrievalConfig,
    SearchResponse,
    SearchService,
)


# ---------------------------------------------------------------- 测试数据

def make_result(doc_id, content="内容", score=0.1, metadata=None):
    """构造 DocumentStore.search 返回的一条结果（score 为余弦距离）。"""
    return {
        "id": doc_id,
        "content": content,
        "metadata": {"namespace": "default"} if metadata is None else metadata,
        "score": score,
    }


RESULTS = [
    make_result("doc_0", "苹果是一种水果。", 0.05, {"category": "fruit"}),
    make_result("doc_1", "香蕉也是一种水果。", 0.30, {"category": "fruit"}),
    make_result("doc_2", "汽车在道路上行驶。", 0.75, {"category": "vehicle"}),
]


def print_response(response):
    print(f"\n  query={response.query!r} total_results={response.total_results} "
          f"retrieval_time_ms={response.retrieval_time_ms}")
    for rank, result in enumerate(response.results, 1):
        print(f"    #{rank} id={result['id']} score={result['score']:.4f} "
              f"metadata={result['metadata']} content={result['content']!r}")


# ---------------------------------------------------------------- 假 DocumentStore

class FakeDocumentStore:
    """DocumentStore 替身，只实现 SearchService 用到的 search / get_stats。

    honor_top_k=False 时忽略 top_k、返回全部结果，
    用于验证 SearchService 自己也会做一次截断。
    delay 用于模拟耗时检索，验证 retrieval_time_ms。
    """

    def __init__(self, results=None, stats=None, delay=0.0, honor_top_k=True):
        self.results = list(results or [])
        self.stats = dict(stats or {
            "count": 0,
            "name": "fake",
            "distance": "cosine",
            "embedding_dim": 3,
            "embedding_model": "fake-embedding",
        })
        self.delay = delay
        self.honor_top_k = honor_top_k
        self.search_calls = []
        self.stats_calls = 0
        self.revision = 0

    def search(self, query, top_k=5, namespace=None, metadata_filter=None):
        self.search_calls.append({
            "query": query,
            "top_k": top_k,
            "namespace": namespace,
            "metadata_filter": metadata_filter,
        })
        if self.delay:
            time.sleep(self.delay)

        results = [dict(result) for result in self.results]
        if namespace is not None:
            results = [r for r in results
                       if r["metadata"].get("namespace") == namespace]
        if metadata_filter:
            results = [
                r for r in results
                if all(r["metadata"].get(key) == value
                       for key, value in metadata_filter.items())
            ]
        return results[:top_k] if self.honor_top_k else results

    def get_stats(self):
        self.stats_calls += 1
        return dict(self.stats)


# ---------------------------------------------------------------- fixtures

@pytest.fixture()
def store():
    return FakeDocumentStore(RESULTS)


@pytest.fixture()
def service(store):
    return SearchService(store)


def unique_collection_name():
    return f"test_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------- 数据类

class TestRetrievalConfig:
    """RetrievalConfig：默认值与独立性"""

    def test_defaults(self):
        config = RetrievalConfig()

        print(f"\n  默认配置：{config}")
        assert config.top_k == 5
        assert config.max_distance is None
        assert config.namespace is None
        assert config.metadata_filter is None

    def test_custom_values(self):
        filter_ = {"category": "fruit"}
        config = RetrievalConfig(top_k=2, max_distance=0.3,
                                 namespace="kb1", metadata_filter=filter_)

        print(f"\n  自定义配置：{config}")
        assert config.top_k == 2
        assert config.max_distance == 0.3
        assert config.namespace == "kb1"
        assert config.metadata_filter == filter_

    def test_instances_do_not_share_defaults(self):
        first = RetrievalConfig()
        second = RetrievalConfig()

        first.metadata_filter = {"category": "fruit"}

        print(f"\n  第一个：{first.metadata_filter}，第二个：{second.metadata_filter}")
        assert second.metadata_filter is None


class TestSearchResponse:
    """SearchResponse：字段可读、可构造"""

    def test_fields(self):
        results = [make_result("doc_0")]

        response = SearchResponse(
            query="水果",
            total_results=1,
            results=results,
            retrieval_time_ms=3,
        )

        print(f"\n  {response}")
        assert response.query == "水果"
        assert response.total_results == 1
        assert response.results == results
        assert response.retrieval_time_ms == 3


# ---------------------------------------------------------------- 初始化

class TestInit:
    """构造与依赖注入"""

    def test_uses_injected_store(self, store, service):
        print(f"\n  store={type(service.store).__name__}")
        assert service.store is store


# ---------------------------------------------------------------- 默认配置与透传

class TestSearchArguments:
    """search：默认配置、显式配置透传"""

    def test_none_config_uses_defaults(self, service, store):
        service.search("水果")

        print(f"\n  传给 DocumentStore.search 的参数：{store.search_calls}")
        assert store.search_calls == [
            {"query": "水果", "top_k": 5, "namespace": None,
             "metadata_filter": None}
        ]

    def test_explicit_config_is_forwarded(self, service, store):
        config = RetrievalConfig(top_k=2, namespace="kb1",
                                 metadata_filter={"category": "fruit"})

        service.search("水果", config=config)

        print(f"\n  传给 DocumentStore.search 的参数：{store.search_calls}")
        call = store.search_calls[0]
        assert call["query"] == "水果"
        assert call["top_k"] == 2
        assert call["namespace"] == "kb1"
        assert call["metadata_filter"] == {"category": "fruit"}

    def test_config_is_not_mutated(self, service):
        config = RetrievalConfig(top_k=2, namespace="kb1")

        service.search("水果", config=config)

        print(f"\n  调用后的配置：{config}")
        assert config == RetrievalConfig(top_k=2, namespace="kb1")


# ---------------------------------------------------------------- 结果结构

class TestResults:
    """search：返回结构、内容透传、空结果"""

    def test_returns_search_response(self, service):
        response = service.search("水果")

        print_response(response)
        assert isinstance(response, SearchResponse)

    def test_result_shape(self, service):
        response = service.search("水果", RetrievalConfig(top_k=10))

        print_response(response)
        for result in response.results:
            assert set(result) == {"id", "content", "metadata", "score"}

    def test_results_are_passed_through(self, service):
        response = service.search("水果", RetrievalConfig(top_k=10))

        print_response(response)
        assert response.results == RESULTS
        assert response.total_results == len(RESULTS)

    def test_query_is_echoed(self, service):
        response = service.search("水果")

        print(f"\n  原始查询=水果，返回 query={response.query!r}")
        assert response.query == "水果"

    def test_empty_results(self):
        service = SearchService(FakeDocumentStore([]))

        response = service.search("不存在的关键词")

        print_response(response)
        assert response.results == []
        assert response.total_results == 0

    def test_each_call_returns_new_response(self, service):
        first = service.search("水果")
        second = service.search("水果")

        print(f"\n  两次返回是同一对象：{first is second}")
        assert first is not second
        assert first == second

    def test_repeated_search_uses_cache_without_sharing_mutable_results(self, store):
        service = SearchService(store)

        first = service.search("水果")
        first.results[0]["content"] = "调用方修改"
        second = service.search("水果")

        assert len(store.search_calls) == 1
        assert first is not second
        assert second.results[0]["content"] == RESULTS[0]["content"]

    def test_store_revision_invalidates_cached_search(self, store):
        service = SearchService(store)

        service.search("水果")
        store.revision += 1
        service.search("水果")

        assert len(store.search_calls) == 2


# ---------------------------------------------------------------- top_k

class TestTopK:
    """search：top_k 截断"""

    def test_top_k_limits_results(self):
        service = SearchService(FakeDocumentStore(RESULTS, honor_top_k=False))

        response = service.search("水果", RetrievalConfig(top_k=2))

        print_response(response)
        assert [r["id"] for r in response.results] == ["doc_0", "doc_1"]
        assert response.total_results == 2

    def test_top_k_larger_than_results(self, service):
        response = service.search("水果", RetrievalConfig(top_k=99))

        print_response(response)
        assert response.total_results == len(RESULTS)

    def test_top_k_zero_returns_empty(self, service):
        response = service.search("水果", RetrievalConfig(top_k=0))

        print_response(response)
        assert response.results == []
        assert response.total_results == 0


# ---------------------------------------------------------------- max_distance

class TestMaxDistance:
    """search：max_distance 距离过滤（保留 score <= max_distance 的结果）"""

    def test_filters_results_above_max_distance(self, service):
        response = service.search("水果", RetrievalConfig(top_k=10, max_distance=0.3))

        print_response(response)
        assert [r["id"] for r in response.results] == ["doc_0", "doc_1"]
        assert response.total_results == 2

    def test_max_distance_boundary_is_inclusive(self, service):
        response = service.search("水果", RetrievalConfig(top_k=10, max_distance=0.30))

        print_response(response)
        assert "doc_1" in [r["id"] for r in response.results]

    def test_max_distance_can_filter_everything(self, service):
        response = service.search("水果", RetrievalConfig(top_k=10, max_distance=-1.0))

        print_response(response)
        assert response.results == []
        assert response.total_results == 0

    def test_max_distance_none_keeps_all(self, service):
        response = service.search("水果", RetrievalConfig(top_k=10))

        print_response(response)
        assert response.total_results == len(RESULTS)

    def test_max_distance_combines_with_top_k(self, service):
        # 距离最近的 doc_0 先被返回，再按 top_k=1 截断
        response = service.search("水果", RetrievalConfig(top_k=1, max_distance=0.3))

        print_response(response)
        assert [r["id"] for r in response.results] == ["doc_0"]
        assert response.total_results == 1


# ---------------------------------------------------------------- metadata_filter

class TestMetadataFilter:
    """search：metadata_filter 过滤与透传"""

    def test_metadata_filter_is_applied(self, service):
        config = RetrievalConfig(top_k=10, metadata_filter={"category": "fruit"})

        response = service.search("水果", config=config)

        print_response(response)
        assert [r["id"] for r in response.results] == ["doc_0", "doc_1"]
        assert response.total_results == 2

    def test_metadata_filter_none_keeps_all(self, service):
        response = service.search("水果", RetrievalConfig(top_k=10))

        print_response(response)
        assert response.total_results == len(RESULTS)

    def test_metadata_filter_is_forwarded(self, service, store):
        config = RetrievalConfig(metadata_filter={"category": "fruit"})

        service.search("水果", config=config)

        print(f"\n  传给 DocumentStore.search 的参数：{store.search_calls}")
        assert store.search_calls[0]["metadata_filter"] == {"category": "fruit"}


# ---------------------------------------------------------------- 计时

class TestRetrievalTime:
    """search：retrieval_time_ms"""

    def test_retrieval_time_is_non_negative_int(self, service):
        response = service.search("水果")

        print(f"\n  retrieval_time_ms={response.retrieval_time_ms} "
              f"type={type(response.retrieval_time_ms).__name__}")
        assert isinstance(response.retrieval_time_ms, int)
        assert response.retrieval_time_ms >= 0

    def test_retrieval_time_measures_elapsed_time(self):
        slow_store = FakeDocumentStore(RESULTS, delay=0.02)
        service = SearchService(slow_store)

        response = service.search("水果")

        print(f"\n  人为延迟 20ms，retrieval_time_ms={response.retrieval_time_ms}")
        assert response.retrieval_time_ms >= 10


# ---------------------------------------------------------------- get_stats

class TestGetStats:
    """get_stats：统计信息透传"""

    def test_returns_store_stats(self, store, service):
        stats = service.get_stats()

        print(f"\n  统计信息：{stats}")
        assert stats == store.stats

    def test_calls_store_once(self, store, service):
        service.get_stats()

        print(f"\n  get_stats 调用次数：{store.stats_calls}")
        assert store.stats_calls == 1


# ---------------------------------------------------------------- 异常

class TestErrorPropagation:
    """底层异常应原样向上抛出"""

    def test_search_error_propagates(self):
        class BrokenStore(FakeDocumentStore):
            def search(self, query, top_k=5, namespace=None, metadata_filter=None):
                raise RuntimeError("向量库不可用")

        service = SearchService(BrokenStore())

        print("\n  期望抛出 RuntimeError")
        with pytest.raises(RuntimeError, match="向量库不可用"):
            service.search("水果")

    def test_get_stats_error_propagates(self):
        class BrokenStore(FakeDocumentStore):
            def get_stats(self):
                raise RuntimeError("统计失败")

        service = SearchService(BrokenStore())

        print("\n  期望抛出 RuntimeError")
        with pytest.raises(RuntimeError, match="统计失败"):
            service.get_stats()


# ---------------------------------------------------------------- 集成测试

class TestIntegration:
    """真实 DocumentStore / ChromaVectorStore 的契约测试"""

    class FakeEmbedding(BaseEmbedding):
        """固定返回 [1, 0, 0]，不加载真实模型。"""

        DIMENSION = 3
        NAME = "fake-embedding"

        def embed_text(self, text):
            return [1.0, 0.0, 0.0]

        def embed_batch(self, texts):
            return [[1.0, 0.0, 0.0] for _ in texts]

        def get_dimension(self):
            return self.DIMENSION

        def get_name(self):
            return self.NAME

    @pytest.fixture()
    def real_store(self, tmp_path):
        vector_store = ChromaVectorStore(
            collection_name=unique_collection_name(),
            persist_directory=str(tmp_path / "chroma_db"),
        )
        return DocumentStore(self.FakeEmbedding(), vector_store=vector_store)

    def test_search_with_real_document_store(self, real_store):
        real_store.add_documents(
            ["苹果是一种水果。", "汽车在道路上行驶。"],
            [{"category": "fruit"}, {"category": "vehicle"}],
            namespace="kb1",
        )
        service = SearchService(real_store)

        response = service.search("水果")

        print_response(response)
        assert response.total_results == 2
        assert {r["content"] for r in response.results} == {
            "苹果是一种水果。", "汽车在道路上行驶。",
        }

    def test_namespace_and_metadata_filter_through_real_store(self, real_store):
        real_store.add_documents(
            ["苹果是一种水果。", "汽车在道路上行驶。"],
            [{"category": "fruit"}, {"category": "vehicle"}],
            namespace="kb1",
        )
        real_store.add_documents(
            ["香蕉也是一种水果。"],
            [{"category": "fruit"}],
            namespace="kb2",
        )
        service = SearchService(real_store)

        response = service.search("水果", RetrievalConfig(
            namespace="kb1", metadata_filter={"category": "fruit"}))

        print_response(response)
        assert [r["content"] for r in response.results] == ["苹果是一种水果。"]
        assert response.results[0]["metadata"]["namespace"] == "kb1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

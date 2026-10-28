"""ChromaVectorStore 测试。

每个用例使用 tmp_path 下的临时持久化目录和随机集合名，互不影响；
向量为手工构造的简单向量，不依赖嵌入模型，结果可重复。
每个用例都会打印实际结果，便于直接观察。

运行方式：
    pytest tests/test_chroma_store.py -v -s
    # 或直接运行
    python tests/test_chroma_store.py
"""

import os
import sys
import uuid

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Windows 控制台默认 GBK，统一切到 UTF-8，避免打印中文时乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

from src.embeddings.chroma_store import ChromaVectorStore
from src.embeddings.vector_store import BaseVectorStore, Document, SearchResult


# ---------------------------------------------------------------- 测试数据

# 查询向量 [1, 0, 0] 下的余弦距离：
#   doc_a = 0  <  doc_b ≈ 0.006  <  doc_d ≈ 0.890  <  doc_c = 1
QUERY = [1.0, 0.0, 0.0]

DOCUMENTS = [
    ("doc_a", "苹果是一种常见的水果。", {"category": "fruit", "source": "a.txt"}, [1.0, 0.0, 0.0]),
    ("doc_b", "香蕉也是水果，富含钾。", {"category": "fruit", "source": "b.txt"}, [0.9, 0.1, 0.0]),
    ("doc_c", "汽车在公路上快速行驶。", {"category": "vehicle", "source": "c.txt"}, [0.0, 1.0, 0.0]),
    ("doc_d", "高铁列车准时到达车站。", {"category": "vehicle", "source": "d.txt"}, [0.1, 0.9, 0.0]),
]

EXPECTED_ORDER = ["doc_a", "doc_b", "doc_d", "doc_c"]


# ---------------------------------------------------------------- 测试辅助

def _unique_collection_name() -> str:
    """生成合法的随机集合名（chroma 要求 3-512 位字母、数字、. _ -）。"""
    return f"test_{uuid.uuid4().hex[:12]}"


def _make_documents(items=DOCUMENTS):
    return [Document(id=doc_id, content=content, metadata=metadata)
            for doc_id, content, metadata, _ in items]


def _make_embeddings(items=DOCUMENTS):
    return [embedding for *_, embedding in items]


def _sample_ids(store) -> list:
    return [result.document.id for result in store.search(QUERY, top_k=10)]


def _print_results(results):
    print(f"\n  共 {len(results)} 条结果（按距离升序）：")
    for rank, result in enumerate(results, 1):
        print(f"    #{rank} id={result.document.id} "
              f"score={result.score:.6f} "
              f"metadata={result.document.metadata} "
              f"content={result.document.content}")


@pytest.fixture()
def persist_dir(tmp_path) -> str:
    return str(tmp_path / "chroma_db")


@pytest.fixture()
def collection_name() -> str:
    return _unique_collection_name()


@pytest.fixture()
def store(persist_dir, collection_name) -> ChromaVectorStore:
    return ChromaVectorStore(collection_name=collection_name, persist_directory=persist_dir)


# ---------------------------------------------------------------- 初始化

class TestInit:
    """初始化与集合管理"""

    def test_implements_base_vector_store(self, store):
        print(f"\n  抽象方法：{ChromaVectorStore.__abstractmethods__}")
        print(f"  是 BaseVectorStore 实例：{isinstance(store, BaseVectorStore)}")
        assert isinstance(store, BaseVectorStore)
        # 基类抽象方法必须全部实现，否则无法实例化
        assert ChromaVectorStore.__abstractmethods__ == frozenset()

    def test_defaults(self, persist_dir):
        store = ChromaVectorStore(persist_directory=persist_dir)

        stats = store.get_collection_stats()
        print(f"\n  默认统计信息：{stats}")
        assert stats["name"] == "rag_documents"
        assert stats["distance"] == "cosine"
        assert stats["count"] == 0

    def test_custom_collection_name_and_distance(self, persist_dir):
        name = _unique_collection_name()
        store = ChromaVectorStore(
            collection_name=name,
            persist_directory=persist_dir,
            distance="l2",
        )

        stats = store.get_collection_stats()
        print(f"\n  自定义统计信息：{stats}")
        print(f"  集合元数据：{store.collection.metadata}")
        assert stats["name"] == name
        assert stats["distance"] == "l2"
        assert store.collection.metadata == {
            "hnsw:space": "l2",
            "hnsw:search_ef": 100,
            "hnsw:construction_ef": 100,
            "hnsw:M": 16,
        }

    def test_reopen_keeps_existing_collection(self, persist_dir, collection_name):
        first = ChromaVectorStore(collection_name=collection_name, persist_directory=persist_dir)
        added = first.add_documents(_make_documents(), _make_embeddings())
        print(f"\n  首次写入 {len(added)} 条：{added}")

        # 同一目录、同一集合名重新打开，数据应保留而不是被重建
        second = ChromaVectorStore(collection_name=collection_name, persist_directory=persist_dir)
        ids = _sample_ids(second)
        print(f"  重新打开后 count={second.get_collection_stats()['count']}")
        print(f"  重新打开后检索顺序：{ids}")
        assert second.get_collection_stats()["count"] == len(DOCUMENTS)
        assert ids == EXPECTED_ORDER


# ---------------------------------------------------------------- 添加文档

class TestAddDocuments:
    """add_documents"""

    def test_returns_ids_and_updates_count(self, store):
        ids = store.add_documents(_make_documents(), _make_embeddings())

        print(f"\n  返回的 ID：{ids}")
        print(f"  集合数量：{store.get_collection_stats()['count']}")
        assert ids == [doc_id for doc_id, *_ in DOCUMENTS]
        assert store.get_collection_stats()["count"] == len(DOCUMENTS)

    def test_generates_id_when_missing(self, store):
        documents = [Document(id="", content="没有 ID 的文档。", metadata={"source": "x.txt"})]

        ids = store.add_documents(documents, [[1.0, 0.0, 0.0]])

        print(f"\n  输入 ID：{documents[0].id!r}")
        print(f"  自动生成 ID：{ids[0]}")
        assert len(ids) == 1
        assert ids[0], "缺少 ID 时应生成一个非空 ID"
        assert uuid.UUID(ids[0]).version == 4
        assert store.get_collection_stats()["count"] == 1

    def test_preserves_content_and_metadata(self, store):
        store.add_documents(_make_documents(), _make_embeddings())

        results = store.search(QUERY, top_k=len(DOCUMENTS))
        by_id = {r.document.id: r.document for r in results}
        print("\n  写入后检索到的内容：")
        for doc_id, content, metadata, _ in DOCUMENTS:
            print(f"    {doc_id}: content={by_id[doc_id].content} metadata={by_id[doc_id].metadata}")
            assert by_id[doc_id].content == content
            assert by_id[doc_id].metadata == metadata

    def test_multiple_calls_accumulate(self, store):
        store.add_documents(_make_documents()[:2], _make_embeddings()[:2])
        count_after_first = store.get_collection_stats()["count"]

        store.add_documents(_make_documents()[2:], _make_embeddings()[2:])
        count_after_second = store.get_collection_stats()["count"]

        print(f"\n  第一次写入后：{count_after_first} 条")
        print(f"  第二次写入后：{count_after_second} 条")
        assert count_after_first == 2
        assert count_after_second == len(DOCUMENTS)

    def test_empty_input_is_noop(self, store):
        ids = store.add_documents([], [])

        print(f"\n  空输入返回：{ids}，集合数量：{store.get_collection_stats()['count']}")
        assert ids == []
        assert store.get_collection_stats()["count"] == 0


# ---------------------------------------------------------------- 检索

class TestSearch:
    """search"""

    def test_returns_search_results(self, store):
        store.add_documents(_make_documents(), _make_embeddings())

        results = store.search(QUERY)

        print(f"\n  查询向量：{QUERY}")
        _print_results(results)
        assert len(results) == len(DOCUMENTS)
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(isinstance(r.document, Document) for r in results)
        assert all(isinstance(r.metadata, dict) for r in results)

    def test_orders_by_distance(self, store):
        store.add_documents(_make_documents(), _make_embeddings())

        results = store.search(QUERY, top_k=len(DOCUMENTS))

        print(f"\n  查询向量：{QUERY}")
        print(f"  期望顺序：{EXPECTED_ORDER}")
        _print_results(results)
        scores = [r.score for r in results]
        assert [r.document.id for r in results] == EXPECTED_ORDER
        assert scores == sorted(scores), "cosine 距离应升序排列"
        assert scores[0] == pytest.approx(0.0, abs=1e-5), "与查询向量相同的文档距离应约等于 0"

    def test_top_k_limits_results(self, store):
        store.add_documents(_make_documents(), _make_embeddings())

        results = store.search(QUERY, top_k=2)

        print(f"\n  top_k=2，期望：{EXPECTED_ORDER[:2]}")
        _print_results(results)
        assert [r.document.id for r in results] == EXPECTED_ORDER[:2]

    def test_filter_by_metadata(self, store):
        store.add_documents(_make_documents(), _make_embeddings())

        filter_ = {"category": "vehicle"}
        results = store.search(QUERY, top_k=10, filter=filter_)

        print(f"\n  过滤条件：{filter_}")
        _print_results(results)
        assert [r.document.id for r in results] == ["doc_d", "doc_c"]
        assert all(r.document.metadata["category"] == "vehicle" for r in results)

    def test_empty_collection_returns_empty_list(self, store):
        results = store.search(QUERY)

        print(f"\n  空集合检索结果：{results}")
        assert results == []


# ---------------------------------------------------------------- 清理与统计

class TestClearAndStats:
    """clear / get_collection_stats"""

    def test_clear_removes_all_documents(self, store):
        store.add_documents(_make_documents(), _make_embeddings())
        before = store.get_collection_stats()["count"]

        cleared = store.clear()
        after = store.get_collection_stats()["count"]
        results = store.search(QUERY)

        print(f"\n  清空前：{before} 条，清空返回：{cleared}，清空后：{after} 条")
        print(f"  清空后检索结果：{results}")
        assert cleared is True
        assert after == 0
        assert results == []

    def test_store_is_usable_after_clear(self, store):
        store.add_documents(_make_documents(), _make_embeddings())
        store.clear()

        ids = store.add_documents(_make_documents()[:1], _make_embeddings()[:1])
        results = store.search(QUERY)

        print(f"\n  清空后重新写入 ID：{ids}")
        _print_results(results)
        assert ids == ["doc_a"]
        assert [r.document.id for r in results] == ["doc_a"]

    def test_collection_stats(self, store):
        store.add_documents(_make_documents(), _make_embeddings())

        stats = store.get_collection_stats()
        print(f"\n  统计信息：{stats}")
        assert stats["count"] == len(DOCUMENTS)
        assert stats["name"] == store.collection_name
        assert stats["distance"] == "cosine"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

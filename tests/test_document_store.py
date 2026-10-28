"""DocumentStore 测试。

覆盖三块能力：
  1. add_parse_result：把 ParseResult 的 chunks 写入向量库，并生成元数据
  2. add_documents：直接追加文本
  3. search / get_stats：命名空间过滤、top_k、统计信息

嵌入模型使用可复现的关键词假向量（不加载真实模型），
向量库使用 tmp_path 下的临时 Chroma 目录，每个用例独立。
每条用例都会打印实际结果，便于直接观察。

运行方式：
    pytest tests/test_document_store.py -v -s
    # 或直接运行
    python tests/test_document_store.py
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

from src.database import document_store as document_store_module
from src.database.document_store import DocumentStore
from src.embeddings.base import BaseEmbedding
from src.embeddings.chroma_store import ChromaVectorStore
from src.parsers.base import ParseResult


# ---------------------------------------------------------------- 假嵌入模型

class FakeEmbedding(BaseEmbedding):
    """按关键词计数生成 3 维向量，结果完全可复现。

    维度约定：
        0 -> 水果类    1 -> 交通类    2 -> 编程类
    """

    DIMENSION = 3
    NAME = "fake-keyword-embedding"
    KEYWORDS = {
        "水果": 0, "苹果": 0, "香蕉": 0,
        "汽车": 1, "道路": 1, "火车": 1,
        "软件": 2, "编程": 2, "模型": 2,
    }

    def __init__(self):
        self.embed_text_calls = []
        self.embed_batch_calls = []

    def _vector(self, text):
        vector = [0.0] * self.DIMENSION
        for word, index in self.KEYWORDS.items():
            vector[index] += text.count(word)
        # 全零向量在余弦距离下没有意义，给一个很小的兜底值
        return vector if any(vector) else [0.01] * self.DIMENSION

    def embed_text(self, text):
        self.embed_text_calls.append(text)
        return self._vector(text)

    def embed_batch(self, texts):
        self.embed_batch_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def get_dimension(self):
        return self.DIMENSION

    def get_name(self):
        return self.NAME


# ---------------------------------------------------------------- 测试数据

FRUIT = "苹果是一种常见的水果。"
BANANA = "香蕉富含钾，也是水果。"
TABLE = "表格（第3页）：\n| 水果 | 价格 |\n| 苹果 | 5 |"
CAR = "汽车在道路上快速行驶。"
SOFTWARE = "软件工程中的编程模型。"


def make_chunk(content, index, chunk_type="text", **extra):
    """按 TextChunker 的输出结构构造一个 chunk。"""
    chunk = {
        "content": content,
        "index": index,
        "length": len(content),
        "type": chunk_type,
    }
    chunk.update(extra)
    return chunk


def make_chunks():
    return [
        make_chunk(FRUIT, 0),
        make_chunk(BANANA, 1),
        make_chunk(TABLE, 2, chunk_type="table", page=3,
                   table_data={"page": 3, "markdown": "| 水果 | 价格 |"}),
        make_chunk(CAR, 3),
    ]


def make_parse_result(file_name="report.pdf", chunks=None, total_pages=1):
    chunks = make_chunks() if chunks is None else chunks
    return ParseResult(
        file_name=file_name,
        total_pages=total_pages,
        total_chunks=len(chunks),
        table_count=sum(1 for chunk in chunks if chunk.get("type") == "table"),
        chunks=chunks,
        raw_text="\n\n".join(chunk["content"] for chunk in chunks),
        tables=[],
        metadata={"parser": "test"},
    )


def expected_ids(parse_result, namespace="default"):
    return [f"{namespace}_{parse_result.file_name}_{chunk['index']}"
            for chunk in parse_result.chunks]


def print_results(results):
    print(f"\n  共 {len(results)} 条结果（按距离升序）：")
    for rank, result in enumerate(results, 1):
        metadata = {k: result["metadata"].get(k) for k in
                    ("file_name", "chunk_index", "chunk_type", "namespace", "page")}
        print(f"    #{rank} id={result['id']} score={result['score']:.6f} "
              f"metadata={metadata} content={result['content']!r}")


# ---------------------------------------------------------------- fixtures

@pytest.fixture()
def embedding():
    return FakeEmbedding()


@pytest.fixture()
def collection_name():
    """chroma 要求集合名 3-512 位，使用随机名避免用例间干扰。"""
    return f"test_{uuid.uuid4().hex[:12]}"


@pytest.fixture()
def vector_store(tmp_path, collection_name):
    return ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=str(tmp_path / "chroma_db"),
    )


@pytest.fixture()
def store(embedding, vector_store):
    return DocumentStore(embedding, vector_store=vector_store)


# ---------------------------------------------------------------- 初始化

class TestInit:
    """构造与依赖注入"""

    def test_uses_injected_dependencies(self, embedding, vector_store, store):
        print(f"\n  embedding={type(store.embedding).__name__} "
              f"vector_store={type(store.vector_store).__name__}")
        assert store.embedding is embedding
        assert store.vector_store is vector_store

    def test_creates_chroma_store_with_collection_name(self, embedding, monkeypatch):
        created = {}

        class RecordingStore:
            def __init__(self, collection_name="rag_documents"):
                created["collection_name"] = collection_name

        monkeypatch.setattr(document_store_module, "ChromaVectorStore", RecordingStore)
        store = DocumentStore(embedding, collection_name="my_docs")

        print(f"\n  未注入 vector_store 时，自动创建参数：{created}")
        assert isinstance(store.vector_store, RecordingStore)
        assert created["collection_name"] == "my_docs"


# ---------------------------------------------------------------- add_parse_result

class TestAddParseResult:
    """add_parse_result：ParseResult -> 向量文档"""

    def test_returns_expected_ids(self, store):
        parse_result = make_parse_result()

        ids = store.add_parse_result(parse_result)

        print(f"\n  返回 ID：{ids}")
        print(f"  期望 ID：{expected_ids(parse_result)}")
        assert ids == expected_ids(parse_result)
        assert store.get_stats()["count"] == len(parse_result.chunks)

    def test_persists_content_and_metadata(self, store):
        parse_result = make_parse_result()
        store.add_parse_result(parse_result)

        results = store.search("水果", top_k=10)
        print_results(results)
        by_id = {result["id"]: result for result in results}

        assert set(by_id) == set(expected_ids(parse_result))
        for chunk in parse_result.chunks:
            doc_id = f"default_{parse_result.file_name}_{chunk['index']}"
            stored = by_id[doc_id]
            print(f"  {doc_id}: metadata={stored['metadata']}")
            assert stored["content"] == chunk["content"]
            assert stored["metadata"]["file_name"] == parse_result.file_name
            assert stored["metadata"]["chunk_index"] == chunk["index"]
            assert stored["metadata"]["chunk_type"] == chunk["type"]
            assert stored["metadata"]["namespace"] == "default"
            assert stored["metadata"]["length"] == len(chunk["content"])

    def test_table_chunk_keeps_page(self, store):
        parse_result = make_parse_result()
        store.add_parse_result(parse_result)

        results = {result["id"]: result for result in store.search("水果", top_k=10)}

        table = results[f"default_{parse_result.file_name}_2"]["metadata"]
        text = results[f"default_{parse_result.file_name}_0"]["metadata"]
        print(f"\n  表格元数据：{table}")
        print(f"  文本元数据：{text}")
        assert table["chunk_type"] == "table"
        assert table["page"] == 3
        # 非表格块不应写入 page
        assert text["chunk_type"] == "text"
        assert "page" not in text

    def test_table_chunk_omits_unknown_page(self, store):
        parse_result = make_parse_result(
            chunks=[make_chunk(TABLE, 0, chunk_type="table", page=None)]
        )

        store.add_parse_result(parse_result)

        result = store.search("水果", top_k=1)[0]
        assert "page" not in result["metadata"]

    def test_embeds_all_contents_in_one_batch(self, store, embedding):
        parse_result = make_parse_result()

        store.add_parse_result(parse_result)

        expected_texts = [chunk["content"] for chunk in parse_result.chunks]
        print(f"\n  embed_batch 调用次数：{len(embedding.embed_batch_calls)}")
        print(f"  传入文本：{embedding.embed_batch_calls}")
        assert embedding.embed_batch_calls == [expected_texts]
        assert embedding.embed_text_calls == []

    def test_file_hash_makes_uploads_idempotent(self, store):
        parse_result = make_parse_result()
        parse_result.metadata["file_hash"] = "a" * 64

        first_ids = store.add_parse_result(parse_result, name_space="kb")
        second_ids = store.add_parse_result(parse_result, name_space="kb")

        assert first_ids == second_ids
        assert store.count_file_hash("a" * 64, "kb") == len(parse_result.chunks)
        assert store.get_stats()["count"] == len(parse_result.chunks)

    def test_custom_namespace_is_recorded(self, store):
        parse_result = make_parse_result()

        ids = store.add_parse_result(parse_result, name_space="kb1")

        print(f"\n  ID：{ids}")
        assert ids == expected_ids(parse_result, namespace="kb1")
        results = store.search("水果", top_k=10, namespace="kb1")
        print_results(results)
        assert all(result["metadata"]["namespace"] == "kb1" for result in results)
        assert store.search("水果", top_k=10, namespace="kb1")

    def test_same_file_in_different_namespaces(self, store):
        parse_result = make_parse_result()

        first = store.add_parse_result(parse_result, name_space="kb1")
        second = store.add_parse_result(parse_result, name_space="kb2")

        print(f"\n  kb1：{first}")
        print(f"  kb2：{second}")
        assert not set(first) & set(second)
        assert store.get_stats()["count"] == 2 * len(parse_result.chunks)

    def test_empty_parse_result(self, store):
        parse_result = make_parse_result(chunks=[])

        ids = store.add_parse_result(parse_result)

        print(f"\n  返回：{ids}，集合数量：{store.get_stats()['count']}")
        assert ids == []
        assert store.get_stats()["count"] == 0


# ---------------------------------------------------------------- search

class TestSearch:
    """search：相似度、top_k、命名空间过滤"""

    def test_returns_expected_shape(self, store):
        store.add_parse_result(make_parse_result())

        results = store.search("水果", top_k=10)

        print_results(results)
        assert len(results) == 4
        for result in results:
            assert set(result) == {"id", "content", "metadata", "score"}
            assert isinstance(result["content"], str)
            assert isinstance(result["metadata"], dict)
            assert result["score"] >= 0

    def test_orders_by_similarity(self, store):
        store.add_parse_result(make_parse_result())

        results = store.search("水果", top_k=10)

        print_results(results)
        scores = [result["score"] for result in results]
        assert scores == sorted(scores), "余弦距离应升序排列"
        # 前三条都与「水果/苹果/香蕉」相关，最后一条是汽车
        assert results[-1]["content"] == CAR
        assert set(result["id"] for result in results[:3]) == {
            "default_report.pdf_0", "default_report.pdf_1", "default_report.pdf_2",
        }

    def test_top_k_limits_results(self, store):
        store.add_parse_result(make_parse_result())

        results = store.search("水果", top_k=2)

        print_results(results)
        assert len(results) == 2
        assert all(result["score"] == pytest.approx(0.0, abs=1e-5) for result in results)

    def test_namespace_filter(self, store):
        store.add_parse_result(
            make_parse_result("fruits.pdf", [make_chunk(FRUIT, 0)]), name_space="fruits")
        store.add_parse_result(
            make_parse_result("cars.pdf", [make_chunk(CAR, 0)]), name_space="cars")

        all_results = store.search("水果", top_k=10)
        fruit_results = store.search("水果", top_k=10, namespace="fruits")
        car_results = store.search("水果", top_k=10, namespace="cars")
        missing_results = store.search("水果", top_k=10, namespace="missing")

        print(f"\n  不带命名空间：{[r['id'] for r in all_results]}")
        print(f"  namespace=fruits：{[r['id'] for r in fruit_results]}")
        print(f"  namespace=cars：{[r['id'] for r in car_results]}")
        print(f"  namespace=missing：{missing_results}")

        assert {r["id"] for r in all_results} == {"fruits_fruits.pdf_0", "cars_cars.pdf_0"}
        assert [r["id"] for r in fruit_results] == ["fruits_fruits.pdf_0"]
        assert [r["id"] for r in car_results] == ["cars_cars.pdf_0"]
        assert missing_results == []

    def test_metadata_filter(self, store):
        store.add_documents(
            [FRUIT, CAR],
            [{"category": "fruit"}, {"category": "vehicle"}],
            namespace="kb",
        )

        filtered = store.search("水果", top_k=10,
                                metadata_filter={"category": "vehicle"})

        print_results(filtered)
        assert [result["content"] for result in filtered] == [CAR]

    def test_namespace_and_metadata_filter_are_combined(self, store):
        store.add_documents([FRUIT], [{"category": "fruit"}], namespace="kb1")
        store.add_documents([CAR], [{"category": "vehicle"}], namespace="kb2")

        results = store.search("水果", top_k=10, namespace="kb1",
                               metadata_filter={"category": "fruit"})

        print_results(results)
        assert [result["metadata"]["namespace"] for result in results] == ["kb1"]
        assert [result["content"] for result in results] == [FRUIT]

    def test_query_is_embedded_once(self, store, embedding):
        store.add_parse_result(make_parse_result())

        store.search("水果")

        print(f"\n  embed_text 调用：{embedding.embed_text_calls}")
        assert embedding.embed_text_calls == ["水果"]

    def test_empty_collection_returns_empty_list(self, store):
        results = store.search("水果")

        print(f"\n  空集合检索结果：{results}")
        assert results == []


# ---------------------------------------------------------------- add_documents

class TestAddDocuments:
    """add_documents：直接追加文本"""

    def test_adds_texts_with_metadata(self, store):
        texts = [FRUIT, CAR]
        metadata = [{"source": "a.txt"}, {"source": "b.txt", "page": 2}]

        ids = store.add_documents(texts, metadata, namespace="kb")

        print(f"\n  返回 ID：{ids}")
        print(f"  集合数量：{store.get_stats()['count']}")
        assert len(ids) == 2
        assert len(set(ids)) == 2
        for doc_id in ids:
            assert doc_id.startswith("kb_")
            assert len(doc_id) == len("kb_") + 8
        assert store.get_stats()["count"] == 2

        results = {r["id"]: r for r in store.search("水果", top_k=10, namespace="kb")}
        by_content = {r["content"]: r for r in results.values()}
        print(f"\n  写入后检索：{ {k: v['metadata'] for k, v in by_content.items()} }")
        assert by_content[FRUIT]["metadata"]["source"] == "a.txt"
        assert by_content[CAR]["metadata"] == {"source": "b.txt", "page": 2, "namespace": "kb"}
        assert all(r["metadata"]["namespace"] == "kb" for r in results.values())

    def test_missing_metadata_entries_default_to_empty(self, store):
        texts = [FRUIT, BANANA, CAR]

        ids = store.add_documents(texts, [{"source": "only-first.txt"}])

        results = store.search("水果", top_k=10)
        by_content = {r["content"]: r["metadata"] for r in results}
        print(f"\n  各文档元数据：{by_content}")
        assert by_content[FRUIT] == {"source": "only-first.txt", "namespace": "default"}
        assert by_content[BANANA] == {"namespace": "default"}
        assert by_content[CAR] == {"namespace": "default"}
        assert len(ids) == 3

    def test_metadata_argument_is_optional(self, store):
        ids = store.add_documents([FRUIT])

        results = store.search("水果", top_k=10)
        print(f"\n  返回：{ids}，元数据：{results[0]['metadata']}")
        assert len(ids) == 1
        assert results[0]["metadata"] == {"namespace": "default"}

    def test_does_not_mutate_caller_metadata(self, store):
        metadata = {"source": "a.txt"}

        store.add_documents([FRUIT], [metadata])

        print(f"\n  调用方原始 metadata：{metadata}")
        assert metadata == {"source": "a.txt"}

    def test_ids_are_unique_across_calls(self, store):
        first = store.add_documents([FRUIT])
        second = store.add_documents([FRUIT])

        print(f"\n  第一次：{first}，第二次：{second}")
        assert not set(first) & set(second)
        assert store.get_stats()["count"] == 2

    def test_empty_input_is_noop(self, store):
        ids = store.add_documents([])

        print(f"\n  返回：{ids}，集合数量：{store.get_stats()['count']}")
        assert ids == []
        assert store.get_stats()["count"] == 0


# ---------------------------------------------------------------- 统计

class TestGetStats:
    """get_stats：向量库统计 + 嵌入模型信息"""

    def test_stats_of_empty_store(self, store, collection_name):
        stats = store.get_stats()

        print(f"\n  空库统计：{stats}")
        assert stats["count"] == 0
        assert stats["name"] == collection_name
        assert stats["distance"] == "cosine"
        assert stats["embedding_dim"] == FakeEmbedding.DIMENSION
        assert stats["embedding_model"] == FakeEmbedding.NAME

    def test_stats_count_reflects_added_documents(self, store):
        parse_result = make_parse_result()
        store.add_parse_result(parse_result)
        store.add_documents([SOFTWARE], namespace="kb")

        stats = store.get_stats()

        print(f"\n  写入后统计：{stats}")
        assert stats["count"] == len(parse_result.chunks) + 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

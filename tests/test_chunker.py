import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.chunker import TextChunker, ChunkStrategy


class TestInit:
    """TextChunker 初始化"""

    def test_default_values(self):
        chunker = TextChunker()
        assert chunker.strategy == ChunkStrategy.PARAGRAPH
        assert chunker.max_chunk_size == 1000
        assert chunker.min_chunk_size == 100
        assert chunker.chunk_overlap == 200

    def test_custom_values(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.SENTENCE,
            max_chunk_size=500,
            min_chunk_size=50,
            chunk_overlap=20,
        )
        assert chunker.strategy == ChunkStrategy.SENTENCE
        assert chunker.max_chunk_size == 500
        assert chunker.min_chunk_size == 50
        assert chunker.chunk_overlap == 20


class TestSplitByParagraph:
    """PARAGRAPH 策略：按空行分段后再按最大块大小合并"""

    def test_empty_text(self):
        assert TextChunker().split("") == []

    def test_whitespace_only(self):
        assert TextChunker().split("\n\n   \n\n") == []

    def test_text_shorter_than_min_chunk_returns_empty(self):
        # 默认 min_chunk_size=100，短文本不会产生分块
        assert TextChunker().split("短文本") == []

    def test_single_paragraph(self):
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=1)
        chunks = chunker.split("hello world")
        assert chunks == [
            {"content": "hello world", "index": 0, "length": 11, "type": "text"}
        ]

    def test_paragraphs_are_combined(self):
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=1, chunk_overlap=0)
        chunks = chunker.split("第一段\n\n第二段")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "第一段\n\n第二段"
        assert chunks[0]["index"] == 0
        assert chunks[0]["type"] == "text"

    def test_split_when_exceeding_max_size(self):
        chunker = TextChunker(max_chunk_size=10, min_chunk_size=1, chunk_overlap=0)
        chunks = chunker.split("abcdef\n\nghijkl")
        assert [c["content"] for c in chunks] == ["abcdef", "ghijkl"]
        assert [c["index"] for c in chunks] == [0, 1]
        assert len(chunks[0]["content"]) <= chunker.max_chunk_size
        assert len(chunks[1]["content"]) <= chunker.max_chunk_size

    def test_overlap_is_kept_between_chunks(self):
        chunker = TextChunker(max_chunk_size=15, min_chunk_size=1, chunk_overlap=5)
        chunks = chunker.split("abcdefghij\n\nklmnopqrst")
        assert [c["content"] for c in chunks] == [
            "abcdefghij",
            "fghij\n\nklmnopqrst",
        ]

    def test_long_paragraph_is_split_by_sentence(self):
        chunker = TextChunker(max_chunk_size=12, min_chunk_size=1, chunk_overlap=0)
        chunks = chunker.split("一二三四五六。七八九十甲乙。")
        assert [c["content"] for c in chunks] == ["一二三四五六。", "七八九十甲乙。"]
        assert [c["index"] for c in chunks] == [0, 1]

    def test_chunk_contains_required_keys(self):
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=1)
        chunk = chunker.split("内容")[0]
        assert set(chunk.keys()) == {"content", "index", "length", "type"}


class TestSplitBySentence:
    """SENTENCE 策略：按中英文句末标点切分"""

    def test_empty_text(self):
        chunker = TextChunker(strategy=ChunkStrategy.SENTENCE, min_chunk_size=1)
        assert chunker.split("") == []

    def test_short_sentence_below_min_chunk_returns_empty(self):
        chunker = TextChunker(strategy=ChunkStrategy.SENTENCE, min_chunk_size=10)
        assert chunker.split("短句。") == []

    def test_chinese_sentences_combined(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.SENTENCE, max_chunk_size=100, min_chunk_size=1
        )
        chunks = chunker.split("第一句。第二句。第三句。")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "第一句。 第二句。 第三句。"
        assert chunks[0]["index"] == 0
        assert chunks[0]["type"] == "text"

    def test_split_when_exceeding_max_size(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.SENTENCE, max_chunk_size=10, min_chunk_size=1
        )
        chunks = chunker.split("一二三四五。六七八九十。")
        assert [c["content"] for c in chunks] == ["一二三四五。", "六七八九十。"]
        assert [c["index"] for c in chunks] == [0, 1]

    def test_english_punctuation(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.SENTENCE, max_chunk_size=100, min_chunk_size=1
        )
        chunks = chunker.split("Hello world. How are you?")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Hello world. How are you?"

    def test_sentence_strategy_differs_from_paragraph(self):
        text = "aaa。\n\nbbb。"
        sentence_chunker = TextChunker(
            strategy=ChunkStrategy.SENTENCE, max_chunk_size=100, min_chunk_size=1
        )
        paragraph_chunker = TextChunker(
            strategy=ChunkStrategy.PARAGRAPH, max_chunk_size=100, min_chunk_size=1
        )
        sentence_chunk = sentence_chunker.split(text)[0]["content"]
        paragraph_chunk = paragraph_chunker.split(text)[0]["content"]
        assert sentence_chunk == "aaa。 bbb。"
        assert "\n\n" in paragraph_chunk


class TestSplitByCharacter:
    """CHARACTER 策略：按固定字符窗口切分，窗口间保留重叠"""

    def test_empty_text(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.CHARACTER, max_chunk_size=5, min_chunk_size=1
        )
        assert chunker.split("") == []

    def test_exact_chunks_without_overlap(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.CHARACTER,
            max_chunk_size=5,
            min_chunk_size=1,
            chunk_overlap=0,
        )
        chunks = chunker.split("abcdefghij")
        assert [c["content"] for c in chunks] == ["abcde", "fghij"]
        assert [c["index"] for c in chunks] == [0, 1]
        assert chunks[0]["length"] == 5
        assert chunks[0]["type"] == "text"

    def test_overlap_between_windows(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.CHARACTER,
            max_chunk_size=5,
            min_chunk_size=2,
            chunk_overlap=2,
        )
        chunks = chunker.split("abcdefghij")
        assert [c["content"] for c in chunks] == ["abcde", "defgh", "ghij"]

    def test_short_tail_below_min_chunk_is_dropped(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.CHARACTER,
            max_chunk_size=5,
            min_chunk_size=5,
            chunk_overlap=0,
        )
        chunks = chunker.split("abcdefgh")
        assert [c["content"] for c in chunks] == ["abcde"]

    def test_content_is_stripped(self):
        chunker = TextChunker(
            strategy=ChunkStrategy.CHARACTER,
            max_chunk_size=5,
            min_chunk_size=1,
            chunk_overlap=0,
        )
        chunks = chunker.split("  abcde")
        # 窗口按原始字符切分（前 5 个字符为 "  abc"），content 会去掉首尾空白
        assert [c["content"] for c in chunks] == ["abc", "de"]
        assert chunks[0]["length"] == 5


class TestStrategyDispatch:
    """split 根据 strategy 选择分块方法"""

    def test_unknown_strategy_falls_back_to_paragraph(self):
        chunker = TextChunker(
            strategy="unknown", max_chunk_size=100, min_chunk_size=1
        )
        chunks = chunker.split("aaa\n\nbbb")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "aaa\n\nbbb"

    def test_default_strategy_is_paragraph(self):
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=1)
        chunks = chunker.split("aaa\n\nbbb")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "aaa\n\nbbb"


class TestAttachTables:
    """split(tables=...) 将表格作为独立的块附加"""

    def test_table_attached_as_separate_chunk(self):
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=1)
        table = {"page": 3, "markdown": "| A |\n|---|\n| 1 |"}
        chunks = chunker.split("正文内容", tables=[table])

        assert len(chunks) == 2
        table_chunk = chunks[1]
        assert table_chunk["index"] == 1
        assert table_chunk["type"] == "table"
        assert table_chunk["page"] == 3
        assert table_chunk["table_data"] == table
        assert table_chunk["content"] == "表格（第3页）：\n| A |\n|---|\n| 1 |"
        assert table_chunk["length"] == len(table_chunk["content"])

    def test_multiple_tables_get_sequential_indexes(self):
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=1)
        tables = [
            {"page": 1, "markdown": "| A |"},
            {"page": 2, "markdown": "| B |"},
        ]
        chunks = chunker.split("正文", tables=tables)
        assert len(chunks) == 3
        assert [c["index"] for c in chunks] == [0, 1, 2]
        assert [c["type"] for c in chunks] == ["text", "table", "table"]

    def test_tables_only_without_text(self):
        chunker = TextChunker(min_chunk_size=1)
        chunks = chunker.split("", tables=[{"page": 1, "markdown": "| A |"}])
        assert len(chunks) == 1
        assert chunks[0]["index"] == 0
        assert chunks[0]["type"] == "table"

    def test_table_without_page_uses_placeholder(self):
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=1)
        chunks = chunker.split("正文", tables=[{"markdown": "| A |"}])
        table_chunk = chunks[1]
        assert table_chunk["page"] is None
        assert table_chunk["content"].startswith("表格（第?页）：")

    def test_empty_tables_does_not_change_result(self):
        chunker = TextChunker(max_chunk_size=100, min_chunk_size=1)
        assert chunker.split("正文", tables=[]) == chunker.split("正文")


if __name__ == "__main__":
    # 无 pytest 时也可直接运行：手动收集并执行所有 test_ 方法
    failures = []
    total = 0
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and name.startswith("Test"):
            for method_name in dir(obj):
                if method_name.startswith("test_"):
                    method = getattr(obj(), method_name)
                    if callable(method):
                        total += 1
                        try:
                            method()
                            print(f"PASS  {name}.{method_name}")
                        except AssertionError as exc:
                            failures.append((name, method_name, exc))
                            print(f"FAIL  {name}.{method_name}: {exc}")

    print(f"\n{total - len(failures)}/{total} 通过")
    if failures:
        raise SystemExit(1)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

from src.parsers.base import ParseResult
from src.parsers.chunker import TextChunker, ChunkStrategy
from src.parsers.cleaner import TextCleaner
from src.parsers.markdown_parser import MarkdownParser


# ---------------------------------------------------------------------------
# 测试辅助：把 markdown 内容写入临时文件，避免依赖 data/ 目录中的样例文件
# ---------------------------------------------------------------------------

def _save_temp_md(content, suffix=".md"):
    """把文本按 utf-8 原样写入临时文件，返回路径（调用方负责删除）"""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    # 用二进制写入，避免 Windows 把 \n 自动转换成 \r\n
    with open(path, "wb") as f:
        f.write(content.encode("utf-8"))
    return path


def _parser(**kwargs):
    """默认使用 min_chunk_size=1 的分块器，保证短样例也能产出分块"""
    kwargs.setdefault("chunker", TextChunker(min_chunk_size=1))
    return MarkdownParser(**kwargs)


SAMPLE_MD = """# 标题

正文第一段内容。

## 小节

正文第二段内容。
"""


class TestSupportedExtensions:
    """supported_extensions 方法"""

    def test_returns_markdown_extensions(self):
        assert MarkdownParser().supported_extensions() == [
            ".md",
            ".MD",
            ".markdown",
            ".MARKDOWN",
        ]


class TestInit:
    """MarkdownParser 初始化：组件可注入，缺省时使用默认实现"""

    def test_default_components(self):
        parser = MarkdownParser()
        assert isinstance(parser.chunker, TextChunker)
        assert isinstance(parser.cleaner, TextCleaner)

    def test_default_chunker_settings(self):
        chunker = MarkdownParser().chunker
        assert chunker.strategy == ChunkStrategy.PARAGRAPH
        assert chunker.max_chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_injected_components_are_kept(self):
        chunker = TextChunker(max_chunk_size=10)
        cleaner = TextCleaner()
        parser = MarkdownParser(chunker=chunker, cleaner=cleaner)
        assert parser.chunker is chunker
        assert parser.cleaner is cleaner


class TestParse:
    """parse 方法：按路径解析 markdown 文件"""

    def test_returns_parse_result(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            result = _parser().parse(path)
            assert isinstance(result, ParseResult)
        finally:
            os.remove(path)

    def test_file_name_is_basename(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            assert _parser().parse(path).file_name == os.path.basename(path)
        finally:
            os.remove(path)

    def test_raw_text_keeps_original_content(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            result = _parser().parse(path)
            assert result.raw_text == SAMPLE_MD
        finally:
            os.remove(path)

    def test_total_pages_is_one(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            assert _parser().parse(path).total_pages == 1
        finally:
            os.remove(path)

    def test_no_tables(self):
        path = _save_temp_md("| a | b |\n|---|---|\n| 1 | 2 |\n")
        try:
            result = _parser().parse(path)
            # markdown 解析器不抽取表格，表格只作为普通文本
            assert result.table_count == 0
            assert result.tables == []
        finally:
            os.remove(path)

    def test_metadata_marks_markdown_parser(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            result = _parser().parse(path)
            assert result.metadata == {
                "parser": "markdown",
                "file_name": os.path.basename(path),
            }
        finally:
            os.remove(path)

    def test_total_chunks_matches_chunks(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            result = _parser().parse(path)
            assert result.total_chunks == len(result.chunks)
            assert result.total_chunks > 0
        finally:
            os.remove(path)

    def test_chunk_indexes_are_sequential(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            chunks = _parser().parse(path).chunks
            assert [c["index"] for c in chunks] == list(range(len(chunks)))
            assert all(c["type"] == "text" for c in chunks)
        finally:
            os.remove(path)

    def test_chunks_contain_all_text_lines(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            contents = "\n".join(c["content"] for c in _parser().parse(path).chunks)
            for line in ["# 标题", "正文第一段内容。", "## 小节", "正文第二段内容。"]:
                assert line in contents
        finally:
            os.remove(path)

    def test_cleaned_text_drops_page_number_lines(self):
        # 清洗器会删除“第 3 页”这类无意义行，原始文本仍保留
        content = "正文内容\n\n第 3 页\n\n更多内容\n"
        path = _save_temp_md(content)
        try:
            result = _parser().parse(path)
            assert "第 3 页" in result.raw_text
            contents = [chunk["content"] for chunk in result.chunks]
            assert not any("第 3 页" in text for text in contents)
        finally:
            os.remove(path)

    def test_empty_file_produces_no_chunks(self):
        path = _save_temp_md("")
        try:
            result = _parser().parse(path)
            assert result.raw_text == ""
            assert result.total_chunks == 0
            assert result.chunks == []
            assert result.tables == []
        finally:
            os.remove(path)

    def test_default_chunker_drops_short_text(self):
        # 默认 min_chunk_size=100，短文档不会产生分块
        path = _save_temp_md("很短的一段话")
        try:
            result = MarkdownParser().parse(path)
            assert result.raw_text == "很短的一段话"
            assert result.chunks == []
        finally:
            os.remove(path)

    def test_custom_cleaner_is_used(self):
        class DummyCleaner(TextCleaner):
            def clean(self, text):
                return "CLEANED"

        path = _save_temp_md("原文")
        try:
            result = _parser(cleaner=DummyCleaner()).parse(path)
            assert result.chunks[0]["content"] == "CLEANED"
        finally:
            os.remove(path)

    def test_missing_file_raises(self):
        try:
            MarkdownParser().parse(os.path.join(tempfile.gettempdir(), "no_such_file.md"))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("文件不存在时应当抛出 FileNotFoundError")


class TestParseFromBytes:
    """parse_from_bytes 方法：从字节流解析 markdown"""

    def test_parses_bytes(self):
        result = _parser().parse_from_bytes(SAMPLE_MD.encode("utf-8"), "sample.md")
        assert result.file_name == "sample.md"
        assert result.raw_text == SAMPLE_MD
        assert result.total_pages == 1
        assert result.total_chunks == len(result.chunks) > 0

    def test_file_name_is_basename(self):
        result = _parser().parse_from_bytes(b"hello", os.path.join("some", "dir", "a.md"))
        assert result.file_name == "a.md"

    def test_matches_parse_result(self):
        path = _save_temp_md(SAMPLE_MD)
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
            parser = _parser()
            from_path = parser.parse(path)
            from_bytes = parser.parse_from_bytes(file_bytes, "sample.md")

            assert from_bytes.chunks == from_path.chunks
            assert from_bytes.tables == from_path.tables
            assert from_bytes.table_count == from_path.table_count
            assert from_bytes.raw_text == from_path.raw_text
        finally:
            os.remove(path)

    def test_crlf_input_yields_same_chunks_as_lf(self):
        # parse 以文本模式读取会把 \r\n 转成 \n，parse_from_bytes 保留 \r\n；
        # 清洗器按行处理，因此分块结果不受影响
        crlf = SAMPLE_MD.replace("\n", "\r\n")
        result = _parser().parse_from_bytes(crlf.encode("utf-8"), "crlf.md")
        lf_result = _parser().parse_from_bytes(SAMPLE_MD.encode("utf-8"), "lf.md")
        assert result.chunks == lf_result.chunks
        assert result.total_chunks == lf_result.total_chunks

    def test_metadata_includes_file_name(self):
        result = _parser().parse_from_bytes(b"# hi", "笔记.md")
        # 与 parse 不同，这里保留调用方传入的完整 file_name
        assert result.metadata == {"parser": "markdown", "file_name": "笔记.md"}

    def test_non_utf8_bytes_raise(self):
        try:
            MarkdownParser().parse_from_bytes(b"\xff\xfe\x00bad", "bad.md")
        except UnicodeDecodeError:
            pass
        else:
            raise AssertionError("非 utf-8 字节流应当抛出 UnicodeDecodeError")


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
                        except Exception as exc:
                            failures.append((name, method_name, exc))
                            print(f"ERROR {name}.{method_name}: {exc!r}")

    print(f"\n{total - len(failures)}/{total} 通过")
    if failures:
        raise SystemExit(1)

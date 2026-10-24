import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

from src.parsers.base import ParseResult
from src.parsers.chunker import TextChunker, ChunkStrategy
from src.parsers.cleaner import TextCleaner
from src.parsers.html_parser import HTMLParser


# ---------------------------------------------------------------------------
# 测试辅助：把 HTML 写入临时文件，避免依赖 data/ 目录中的样例文件
# ---------------------------------------------------------------------------

def _save_temp_html(content, suffix=".html"):
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
    return HTMLParser(**kwargs)


SAMPLE_HTML = """<html><body>
<h1>主标题</h1>
<p>这是第一段正文。</p>
<h2>小节</h2>
<ul><li>要点一</li><li>要点二</li></ul>
<p><a href="https://example.com">链接文字</a></p>
<table><tr><th>名称</th><th>数量</th></tr><tr><td>苹果</td><td>3</td></tr></table>
</body></html>"""


class TestSupportedExtensions:
    """supported_extensions 方法"""

    def test_returns_html_extensions(self):
        assert HTMLParser().supported_extensions() == [
            ".html",
            ".htm",
            ".HTML",
            ".HTM",
        ]


class TestInit:
    """HTMLParser 初始化：组件可注入，缺省时使用默认实现"""

    def test_default_components(self):
        parser = HTMLParser()
        assert isinstance(parser.chunker, TextChunker)
        assert isinstance(parser.cleaner, TextCleaner)

    def test_default_chunker_settings(self):
        chunker = HTMLParser().chunker
        assert chunker.strategy == ChunkStrategy.PARAGRAPH
        assert chunker.max_chunk_size == 1000
        assert chunker.chunk_overlap == 200

    def test_injected_components_are_kept(self):
        chunker = TextChunker(max_chunk_size=10)
        cleaner = TextCleaner()
        parser = HTMLParser(chunker=chunker, cleaner=cleaner)
        assert parser.chunker is chunker
        assert parser.cleaner is cleaner


class TestParse:
    """parse 方法：按路径解析 HTML 文件"""

    def test_returns_parse_result(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            result = _parser().parse(path)
            assert isinstance(result, ParseResult)
        finally:
            os.remove(path)

    def test_file_name_is_basename(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            assert _parser().parse(path).file_name == os.path.basename(path)
        finally:
            os.remove(path)

    def test_total_pages_is_one(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            assert _parser().parse(path).total_pages == 1
        finally:
            os.remove(path)

    def test_no_tables(self):
        # HTML 解析器不抽取表格，<table> 只作为普通 markdown 文本保留
        path = _save_temp_html(SAMPLE_HTML)
        try:
            result = _parser().parse(path)
            assert result.table_count == 0
            assert result.tables == []
        finally:
            os.remove(path)

    def test_metadata_marks_markdownify_parser(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            assert _parser().parse(path).metadata == {"parser": "markdownify"}
        finally:
            os.remove(path)

    def test_raw_text_is_markdownified_and_cleaned(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            result = _parser().parse(path)
            # HTML 标签被转换：标题变成纯文本，链接变成 markdown 链接，表格变成管道表格
            assert "<h1>" not in result.raw_text
            assert "主标题" in result.raw_text
            assert "[链接文字](https://example.com)" in result.raw_text
            assert "| 名称 | 数量 |" in result.raw_text
            # 清洗器会删除 setext 标题下划线（=== / ---）
            assert "===" not in result.raw_text
        finally:
            os.remove(path)

    def test_total_chunks_matches_chunks(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            result = _parser().parse(path)
            assert result.total_chunks == len(result.chunks)
            assert result.total_chunks > 0
        finally:
            os.remove(path)

    def test_chunk_indexes_are_sequential(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            chunks = _parser().parse(path).chunks
            assert [c["index"] for c in chunks] == list(range(len(chunks)))
            assert all(c["type"] == "text" for c in chunks)
        finally:
            os.remove(path)

    def test_chunks_contain_all_text_lines(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            contents = "\n".join(c["content"] for c in _parser().parse(path).chunks)
            for line in [
                "主标题",
                "这是第一段正文。",
                "小节",
                "* 要点一",
                "* 要点二",
                "[链接文字](https://example.com)",
                "| 苹果 | 3 |",
            ]:
                assert line in contents
        finally:
            os.remove(path)

    def test_english_text_is_preserved(self):
        # 清洗只删除无意义行，不应丢掉非中文内容
        path = _save_temp_html("<p>Hello world</p>")
        try:
            assert "Hello world" in _parser().parse(path).raw_text
        finally:
            os.remove(path)

    def test_cleaned_text_drops_page_number_lines(self):
        path = _save_temp_html("<p>正文内容</p><p>第 3 页</p><p>更多内容</p>")
        try:
            result = _parser().parse(path)
            assert "第 3 页" not in result.raw_text
            contents = [chunk["content"] for chunk in result.chunks]
            assert not any("第 3 页" in text for text in contents)
        finally:
            os.remove(path)

    def test_empty_file_produces_no_chunks(self):
        path = _save_temp_html("")
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
        path = _save_temp_html("<p>很短的一段话</p>")
        try:
            result = HTMLParser().parse(path)
            assert result.raw_text == "很短的一段话"
            assert result.chunks == []
        finally:
            os.remove(path)

    def test_custom_cleaner_is_used(self):
        class DummyCleaner(TextCleaner):
            def clean(self, text):
                return "CLEANED"

        path = _save_temp_html("<p>原文</p>")
        try:
            result = _parser(cleaner=DummyCleaner()).parse(path)
            assert result.chunks[0]["content"] == "CLEANED"
        finally:
            os.remove(path)

    def test_missing_file_raises(self):
        try:
            HTMLParser().parse(os.path.join(tempfile.gettempdir(), "no_such_file.html"))
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("文件不存在时应当抛出 FileNotFoundError")


class TestParseFromBytes:
    """parse_from_bytes 方法：从字节流解析 HTML"""

    def test_parses_bytes(self):
        result = _parser().parse_from_bytes(SAMPLE_HTML.encode("utf-8"), "sample.html")
        assert result.file_name == "sample.html"
        assert "主标题" in result.raw_text
        assert result.total_pages == 1
        assert result.total_chunks == len(result.chunks) > 0

    def test_file_name_is_basename(self):
        result = _parser().parse_from_bytes(
            "<p>hi</p>".encode("utf-8"), os.path.join("some", "dir", "a.html")
        )
        assert result.file_name == "a.html"

    def test_matches_parse_result(self):
        path = _save_temp_html(SAMPLE_HTML)
        try:
            with open(path, "rb") as f:
                file_bytes = f.read()
            parser = _parser()
            from_path = parser.parse(path)
            from_bytes = parser.parse_from_bytes(file_bytes, "sample.html")

            assert from_bytes.chunks == from_path.chunks
            assert from_bytes.tables == from_path.tables
            assert from_bytes.table_count == from_path.table_count
            assert from_bytes.raw_text == from_path.raw_text
        finally:
            os.remove(path)

    def test_non_utf8_bytes_raise(self):
        try:
            HTMLParser().parse_from_bytes(b"\xff\xfe\x00bad", "bad.html")
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

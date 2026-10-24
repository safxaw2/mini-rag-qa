import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile

from src.parsers import ParserFactory, parse_file
from src.parsers.base import BaseParser, ParseResult
from src.parsers.html_parser import HTMLParser
from src.parsers.markdown_parser import MarkdownParser
from src.parsers.word_paser import Wordarser


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------

def _save_temp(content, suffix):
    """把文本按 utf-8 原样写入临时文件，返回路径（调用方负责删除）"""
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        f.write(content.encode("utf-8"))
    return path


def _restore_registry(snapshot):
    """恢复工厂注册表：register_parser 会修改类级字典，测试后必须还原"""
    ParserFactory._parsers.clear()
    ParserFactory._parsers.update(snapshot)


class DummyParser(BaseParser):
    """用于测试 register_parser 的最小解析器"""

    def parse(self, file_path: str) -> ParseResult:
        return ParseResult(
            file_name=os.path.basename(file_path),
            total_pages=1,
            total_chunks=0,
            table_count=0,
            chunks=[],
            raw_text="DUMMY",
            tables=[],
            metadata={"parser": "dummy"},
        )

    def parse_from_bytes(self, file_bytes: bytes, file_name: str) -> ParseResult:
        return self.parse(file_name)

    def supported_extensions(self) -> list[str]:
        return [".dummy"]


class TestSupportedExtensions:
    """ParserFactory.supported_extensions 方法"""

    def test_lists_all_supported_formats(self):
        assert set(ParserFactory.supported_extensions()) == {
            ".pdf",
            ".html",
            ".htm",
            ".md",
            ".markdown",
            ".docx",
        }

    def test_extensions_are_dotted_and_lowercase(self):
        # get_parser 用 Path(...).suffix.lower() 查表，键必须带点且小写
        for ext in ParserFactory.supported_extensions():
            assert ext.startswith(".")
            assert ext == ext.lower()

    def test_returns_a_copy(self):
        # 修改返回值不应影响工厂注册表
        extensions = ParserFactory.supported_extensions()
        extensions.append(".fake")
        assert ".fake" not in ParserFactory.supported_extensions()


class TestGetParser:
    """ParserFactory.get_parser 方法：按扩展名分派"""

    def test_markdown_parser(self):
        assert isinstance(ParserFactory.get_parser("note.md"), MarkdownParser)

    def test_markdown_alias(self):
        assert isinstance(ParserFactory.get_parser("note.markdown"), MarkdownParser)

    def test_html_parser(self):
        assert isinstance(ParserFactory.get_parser("page.html"), HTMLParser)

    def test_htm_alias(self):
        assert isinstance(ParserFactory.get_parser("page.htm"), HTMLParser)

    def test_docx_parser(self):
        assert isinstance(ParserFactory.get_parser("doc.docx"), Wordarser)

    def test_extension_is_case_insensitive(self):
        assert isinstance(ParserFactory.get_parser("NOTE.MD"), MarkdownParser)
        assert isinstance(ParserFactory.get_parser("PAGE.HTML"), HTMLParser)
        assert isinstance(ParserFactory.get_parser("DOC.DOCX"), Wordarser)

    def test_path_with_directories(self):
        path = os.path.join("data", "sub", "报告.md")
        assert isinstance(ParserFactory.get_parser(path), MarkdownParser)

    def test_returns_new_instance_each_call(self):
        assert ParserFactory.get_parser("a.md") is not ParserFactory.get_parser("a.md")

    def test_unsupported_extension_raises(self):
        try:
            ParserFactory.get_parser("data.txt")
        except ValueError as exc:
            assert ".txt" in str(exc)
        else:
            raise AssertionError("不支持的格式应当抛出 ValueError")

    def test_no_extension_raises(self):
        try:
            ParserFactory.get_parser("README")
        except ValueError:
            pass
        else:
            raise AssertionError("没有扩展名时应当抛出 ValueError")


class TestRegisterParser:
    """ParserFactory.register_parser 方法：运行期注册新格式"""

    def _with_snapshot(self):
        # 注册表是类级共享状态，所有注册测试都要在 finally 中还原
        return dict(ParserFactory._parsers)

    def test_register_class(self):
        snapshot = self._with_snapshot()
        try:
            ParserFactory.register_parser(".dummy", DummyParser)
            assert ".dummy" in ParserFactory.supported_extensions()
            assert isinstance(ParserFactory.get_parser("a.dummy"), DummyParser)
        finally:
            _restore_registry(snapshot)

    def test_register_instance_returns_same_instance(self):
        snapshot = self._with_snapshot()
        try:
            instance = DummyParser()
            ParserFactory.register_parser(".dummy", instance)
            assert ParserFactory.get_parser("a.dummy") is instance
        finally:
            _restore_registry(snapshot)

    def test_extension_is_normalized(self):
        snapshot = self._with_snapshot()
        try:
            ParserFactory.register_parser("DUMMY", DummyParser)
            assert ".dummy" in ParserFactory.supported_extensions()
            assert ".DUMMY" not in ParserFactory.supported_extensions()
            assert isinstance(ParserFactory.get_parser("a.Dummy"), DummyParser)
        finally:
            _restore_registry(snapshot)

    def test_register_overrides_existing_extension(self):
        snapshot = self._with_snapshot()
        try:
            ParserFactory.register_parser(".md", DummyParser)
            assert isinstance(ParserFactory.get_parser("a.md"), DummyParser)
        finally:
            _restore_registry(snapshot)

    def test_registry_is_restored_after_tests(self):
        # 依赖上面各测试的 finally 分支：默认注册表保持不变
        assert isinstance(ParserFactory.get_parser("a.md"), MarkdownParser)
        assert ".dummy" not in ParserFactory.supported_extensions()


class TestParseFile:
    """parse_file 函数：自动识别类型并解析"""

    def test_parses_markdown_file(self):
        path = _save_temp("# 标题\n\n正文内容。\n", ".md")
        try:
            result = parse_file(path)
            assert isinstance(result, ParseResult)
            assert result.file_name == os.path.basename(path)
            assert "正文内容。" in result.raw_text
        finally:
            os.remove(path)

    def test_parses_html_file(self):
        path = _save_temp("<html><body><p>Hello 内容</p></body></html>", ".html")
        try:
            result = parse_file(path)
            assert isinstance(result, ParseResult)
            assert result.file_name == os.path.basename(path)
            assert "Hello 内容" in result.raw_text
        finally:
            os.remove(path)

    def test_uppercase_extension(self):
        path = _save_temp("# 标题\n", ".MD")
        try:
            assert isinstance(parse_file(path), ParseResult)
        finally:
            os.remove(path)

    def test_unsupported_extension_raises(self):
        path = _save_temp("plain text", ".txt")
        try:
            try:
                parse_file(path)
            except ValueError:
                pass
            else:
                raise AssertionError("不支持的格式应当抛出 ValueError")
        finally:
            os.remove(path)

    def test_registered_parser_is_used(self):
        snapshot = dict(ParserFactory._parsers)
        path = _save_temp("anything", ".dummy")
        try:
            ParserFactory.register_parser(".dummy", DummyParser)
            result = parse_file(path)
            assert result.raw_text == "DUMMY"
            assert result.metadata == {"parser": "dummy"}
        finally:
            _restore_registry(snapshot)
            os.remove(path)


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

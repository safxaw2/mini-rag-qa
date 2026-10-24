import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.cleaner import TextCleaner


class TestClean:
    """clean 方法：按行移除无意义的模式并去除空白行"""

    def test_remove_chinese_page_marker(self):
        cleaner = TextCleaner()
        text = "第1页\n这是正文。\n第2页\n"
        result = cleaner.clean(text)
        assert result == "这是正文。"

    def test_remove_english_page_marker(self):
        cleaner = TextCleaner()
        text = "Page 1\nHello world.\nPage 2\n"
        result = cleaner.clean(text)
        assert result == "Hello world."

    def test_remove_standalone_page_number(self):
        cleaner = TextCleaner()
        text = "12\n正文内容\n13\n"
        result = cleaner.clean(text)
        assert result == "正文内容"

    def test_remove_page_number_format(self):
        cleaner = TextCleaner()
        text = "1/10\n内容\n2/10\n"
        result = cleaner.clean(text)
        assert result == "内容"

    def test_remove_separator_line(self):
        cleaner = TextCleaner()
        text = "标题\n--------\n内容\n======\n结尾"
        result = cleaner.clean(text)
        assert result == "标题\n内容\n结尾"

    def test_keep_normal_content(self):
        cleaner = TextCleaner()
        text = "第一行\n第二行\n第三行"
        result = cleaner.clean(text)
        assert result == "第一行\n第二行\n第三行"

    def test_skip_blank_lines(self):
        cleaner = TextCleaner()
        text = "第一行\n\n\n第二行\n"
        result = cleaner.clean(text)
        assert result == "第一行\n第二行"

    def test_strip_surrounding_whitespace(self):
        cleaner = TextCleaner()
        text = "  内容前后有空格  \n"
        result = cleaner.clean(text)
        assert result == "内容前后有空格"

    def test_custom_patterns(self):
        cleaner = TextCleaner(custom_patterns=[r"^\s*[【].*[】]\s*$"])
        text = "【广告】\n正文内容\n【免责声明】\n"
        result = cleaner.clean(text)
        assert result == "正文内容"

    def test_empty_text(self):
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean("   \n\n  ") == ""


class TestNormalizeWhitespace:
    """normalize_withespace 方法：去除多余空白"""

    def test_collapse_multiple_spaces(self):
        cleaner = TextCleaner()
        assert cleaner.normalize_withespace("a   b\t\tc") == "a b c"

    def test_collapse_newlines(self):
        # 只把 3 个以上连续换行压成一个空行，段落结构保留
        cleaner = TextCleaner()
        assert (
            cleaner.normalize_withespace("第一行\n\n第二行\n\n\n第三行")
            == "第一行\n\n第二行\n\n第三行"
        )

    def test_trailing_spaces_before_newline_removed(self):
        cleaner = TextCleaner()
        assert cleaner.normalize_withespace("第一行   \n第二行") == "第一行\n第二行"

    def test_strip_leading_trailing(self):
        cleaner = TextCleaner()
        assert cleaner.normalize_withespace("  \n  内容  \n  ") == "内容"

    def test_empty_text(self):
        cleaner = TextCleaner()
        assert cleaner.normalize_withespace("   \n  ") == ""


class TestRemoveExtraSpaces:
    """remove_extra_spaces 方法：压缩连续的水平空白字符"""

    def test_non_chinese_is_kept(self):
        cleaner = TextCleaner()
        assert cleaner.remove_extra_spaces("hello世界123") == "hello世界123"

    def test_punctuation_is_kept(self):
        cleaner = TextCleaner()
        assert cleaner.remove_extra_spaces("你好，世界！") == "你好，世界！"

    def test_collapse_spaces_and_tabs(self):
        cleaner = TextCleaner()
        assert cleaner.remove_extra_spaces("a   b\t\tc") == "a b c"

    def test_pure_chinese_unchanged(self):
        cleaner = TextCleaner()
        assert cleaner.remove_extra_spaces("纯中文内容") == "纯中文内容"

    def test_empty_text(self):
        cleaner = TextCleaner()
        assert cleaner.remove_extra_spaces("") == ""


class TestFullClean:
    """full_clean 方法：clean -> normalize_withespace -> remove_extra_spaces"""

    def test_pipeline(self):
        cleaner = TextCleaner()
        text = "第1页\n\n这是测试文本。\n\n第2页\n\nPage 3\n"
        result = cleaner.full_clean(text)
        assert "第1页" not in result
        assert "第2页" not in result
        assert "Page" not in result
        assert "这是测试文本" in result

    def test_empty_text(self):
        cleaner = TextCleaner()
        assert cleaner.full_clean("") == ""


if __name__ == "__main__":
    # 无 pytest 时也可直接运行：手动收集并执行所有 test_ 方法
    import types

    failures = []
    total = 0
    for name, obj in list(globals().items()):
        if isinstance(obj, type) and name.startswith("Test"):
            for method_name in dir(obj):
                if method_name.startswith("test_"):
                    method = getattr(obj, method_name)
                    if isinstance(method, types.FunctionType):
                        total += 1
                        try:
                            method(obj())
                            print(f"PASS  {name}.{method_name}")
                        except AssertionError as exc:
                            failures.append((name, method_name, exc))
                            print(f"FAIL  {name}.{method_name}: {exc}")

    print(f"\n{total - len(failures)}/{total} 通过")
    if failures:
        raise SystemExit(1)

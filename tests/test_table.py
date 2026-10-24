import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.parsers.table_parser import TableProcessor


class TestMarkdownToDataframe:
    """markdown_to_dataframe 方法：markdown 表格 -> DataFrame"""

    def test_basic_table(self):
        tp = TableProcessor()
        md = "| Name | Age |\n|------|-----|\n| John | 25  |\n| Jane | 30  |"
        df = tp.markdown_to_dataframe(md)
        assert df.columns.tolist() == ["Name", "Age"]
        assert df.values.tolist() == [["John", "25"], ["Jane", "30"]]

    def test_alignment_separator(self):
        tp = TableProcessor()
        md = "| Name | Age |\n|:-----|----:|\n| John | 25  |"
        df = tp.markdown_to_dataframe(md)
        assert df.columns.tolist() == ["Name", "Age"]
        assert df.values.tolist() == [["John", "25"]]

    def test_no_separator_line(self):
        tp = TableProcessor()
        md = "| A | B |\n| 1 | 2 |"
        df = tp.markdown_to_dataframe(md)
        assert df.columns.tolist() == ["A", "B"]
        assert df.values.tolist() == [["1", "2"]]

    def test_extra_pipes_and_spaces(self):
        tp = TableProcessor()
        md = "  | A | B |  \n| --- | --- |\n| 1 | 2 |"
        df = tp.markdown_to_dataframe(md)
        assert df.columns.tolist() == ["A", "B"]
        assert df.values.tolist() == [["1", "2"]]

    def test_empty_string(self):
        tp = TableProcessor()
        df = tp.markdown_to_dataframe("")
        assert df.empty


class TestDataframeToText:
    """dataframe_to_text 方法：DataFrame -> markdown 文本"""

    def test_output_contains_cells(self):
        tp = TableProcessor()
        df = tp.markdown_to_dataframe("| Name | Age |\n|---|---|\n| John | 25 |")
        text = tp.dataframe_to_text(df)
        assert isinstance(text, str)
        assert "Name" in text
        assert "John" in text
        assert "25" in text

    def test_empty_dataframe(self):
        tp = TableProcessor()
        assert tp.dataframe_to_text(pd.DataFrame()) == ""


class TestTableToText:
    """table_to_text 方法：markdown 表格 -> 文本"""

    def test_round_trip(self):
        tp = TableProcessor()
        md = "| Name | Age |\n|---|---|\n| John | 25 |"
        text = tp.table_to_text(md)
        assert "Name" in text
        assert "John" in text

    def test_empty_text(self):
        tp = TableProcessor()
        assert tp.table_to_text("") == ""


class TestMergeTables:
    """merge_tables 方法：合并多个表格为每行字典列表"""

    def test_same_schema(self):
        tp = TableProcessor()
        tables = [
            {"markdown": "| A | B |\n|---|---|\n| 1 | 2 |"},
            {"markdown": "| A | B |\n|---|---|\n| 3 | 4 |"},
        ]
        result = tp.merge_tables(tables)
        assert result == [{"A": "1", "B": "2"}, {"A": "3", "B": "4"}]

    def test_different_schema_fills_empty(self):
        tp = TableProcessor()
        tables = [
            {"markdown": "| A | B |\n|---|---|\n| 1 | 2 |"},
            {"markdown": "| A | C |\n|---|---|\n| 3 | 4 |"},
        ]
        result = tp.merge_tables(tables)
        assert result == [
            {"A": "1", "B": "2", "C": ""},
            {"A": "3", "B": "", "C": "4"},
        ]

    def test_dict_with_extra_keys(self):
        tp = TableProcessor()
        tables = [
            {"table_number": 1, "page": 1, "markdown": "| A |\n|---|\n| 1 |"},
        ]
        result = tp.merge_tables(tables)
        assert result == [{"A": "1"}]

    def test_empty_list(self):
        tp = TableProcessor()
        assert tp.merge_tables([]) == []


class TestExtractKeyValue:
    """extract_key_value 方法：两列表格 -> 键值对"""

    def test_two_columns(self):
        tp = TableProcessor()
        md = "| Param | Value |\n|-------|-------|\n| name  | Alice |\n| age   | 25    |"
        assert tp.extract_key_value(md) == {"name": "Alice", "age": "25"}

    def test_single_column(self):
        tp = TableProcessor()
        md = "| single |\n|--------|\n| 1      |"
        assert tp.extract_key_value(md) == {}

    def test_empty_text(self):
        tp = TableProcessor()
        assert tp.extract_key_value("") == {}


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

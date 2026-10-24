import re
from typing import List, Dict

import pandas as pd


class TableProcessor:
    '''
    表格解析
    '''

    # 匹配markdown表格分隔行单元格，如 ---、:---、---:、:---:
    _SEPARATOR_RE = re.compile(r':?-{3,}:?')

    def markdown_to_dataframe(self, table_markdown: str) -> pd.DataFrame:
        '''
        将markdown表格转换为DataFrame
        参数：
            table_markdown: markdown表格文本
        返回：
            pd.DataFrame:转换后的表格
        '''
        lines = [line.strip() for line in table_markdown.strip().splitlines() if line.strip()]
        if not lines:
            return pd.DataFrame()

        # 去掉首尾的竖线，再按竖线切分单元格
        rows = [
            [cell.strip() for cell in line.strip().strip('|').split('|')]
            for line in lines
        ]
        header = rows[0]

        # 过滤掉分隔行（如 |---|---|）
        data_rows = [
            row for row in rows[1:]
            if not (row and all(self._SEPARATOR_RE.fullmatch(cell) for cell in row if cell))
        ]

        # 对齐列数，防止畸形表格导致列数不一致
        n_cols = len(header)
        data_rows = [(row + [''] * n_cols)[:n_cols] for row in data_rows]

        return pd.DataFrame(data_rows, columns=header)

    def dataframe_to_text(self, df: pd.DataFrame) -> str:
        '''
        将DataFrame转换为文本
        参数：
            df: DataFrame表格
        返回：
            str: 转换后的文本
        '''
        if df.empty:
            return ''
        return df.to_markdown(index=False)

    def table_to_text(self, table_markdown: str) -> str:
        '''
        将表格转换为文本
        参数：
            table_markdown: markdown表格文本
        返回：
            str: 转换后的文本
        '''
        df = self.markdown_to_dataframe(table_markdown)
        return self.dataframe_to_text(df)

    def merge_tables(self, tables: List[Dict]) -> List[Dict]:
        '''
        将多个表格合并为一个表格
        参数：
            tables: 表格字典列表，每个元素需包含'markdown'键
        返回：
            List[Dict]: 合并后每行字典的列表
        '''
        dfs = []
        for table in tables:
            markdown = table.get('markdown') if isinstance(table, dict) else table
            if markdown:
                dfs.append(self.markdown_to_dataframe(markdown))
        if not dfs:
            return []
        merged_df = pd.concat(dfs, ignore_index=True).fillna('')
        return merged_df.to_dict(orient='records')

    def extract_key_value(self, table_text: str) -> Dict[str, str]:
        '''
        从表格中提取键值对
        参数：
            table_text: markdown表格文本
        返回：
            Dict[str,str]: 键值对
        '''
        df = self.markdown_to_dataframe(table_text)
        if df.empty or df.shape[1] < 2:
            return {}
        return {
            str(row.iloc[0]): str(row.iloc[1])
            for _, row in df.iterrows()
        }

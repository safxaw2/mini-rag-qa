import os
# 必须在导入 docling / huggingface 相关库之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 可选：如果你希望模型缓存到指定磁盘，可以顺便设置
# os.environ["HF_HOME"] = "E:/code/rag_project/models_cache"

from pathlib import Path
from typing import List,Dict,Optional

import docx

from .base import BaseParser,ParseResult
from src.parsers.chunker import TextChunker
from src.parsers.cleaner import TextCleaner
from src.parsers.table_parser import TableProcessor


class Wordarser(BaseParser):
    def __init__(
            self,
            chunker:Optional[TextChunker]=None,
            cleaner:Optional[TextCleaner]=None,
            table_processor:Optional[TableProcessor]=None
    ):

           #初始化组件
           self.chunker=chunker if chunker  else TextChunker()
           self.cleaner=cleaner if cleaner else TextCleaner()
           self.table_processor=table_processor if table_processor else TableProcessor()

    def parse(self,file_path:str) -> ParseResult:
          '''
          通过路径解析文件
          参数：
          file_path:文件路径
          '''
          #创建doc对象
          doc=docx.Document(file_path)

          #提取数据
          paragraphs=[para.text for para in doc.paragraphs if para.text.strip()]
          raw_text='\n\n'.join(paragraphs)

          #提取表格
          table=self._extract_tables(doc)

          #处理数据
          cleaned_text=self.cleaner.clean(raw_text)
          marged_tables=self.table_processor.merge_tables(table)
          chunks=self.chunker.split(cleaned_text,marged_tables)

          return ParseResult(
              file_name=Path(file_path).name,
              total_pages=len(doc.paragraphs),
              total_chunks=len(chunks),
              table_count=len(table),
              chunks=chunks,
              raw_text=raw_text,
              tables=table,
              metadata={'parser':'python-docx'}
          )


    def parse_from_bytes(self,file_bytes,file_name) -> ParseResult:
          '''
          通过字节流解析文件
          参数：
          file_bytes:字节流
          file_name:文件名
          '''

          from io import BytesIO
          stream=BytesIO(file_bytes)
          doc=docx.Document(stream)

          #提取数据
          paragraphs=[para.text for para in doc.paragraphs if para.text.strip()]
          raw_text='\n\n'.join(paragraphs)

          #提取表格
          table=self._extract_tables(doc)

          #处理数据
          cleaned_text=self.cleaner.clean(raw_text)
          marged_tables=self.table_processor.merge_tables(table)
          chunks=self.chunker.split(cleaned_text,marged_tables)

          return ParseResult(
              file_name=file_name,
              total_pages=len(doc.paragraphs),
              total_chunks=len(chunks),
              table_count=len(table),
              chunks=chunks,
              raw_text=raw_text,
              tables=table,
              metadata={'parser':'python-docx','file_name':file_name}
          )

    def supported_extensions(self) -> list[str]:
          '''
          返回支持的文件扩展名
          '''
          return ['.docx','.DOCX']

    def _extract_tables(self,doc) -> List[Dict]:
          '''
          从docx文件中提取表格
          参数：
          doc:docx文件对象
          返回：
          表格列表
          '''
          tables=[]
          for table_index,table in enumerate(doc.tables):
                #转换为markdown
                md_rows=[]
                for row_index,row in enumerate(table.rows):
                      cells=[cell.text.strip() for cell in row.cells]
                      md_rows.append('|'+'|'.join(cells)+'|')
                      #添加表头
                      if row_index==0:
                          md_rows.append('|'+'|'.join(['---']*len(cells))+'|')
                tables.append({
                        'page':1,
                        'row':len(table.rows),
                        'cols':len(table.columns),
                        'markdown':'\n'.join(md_rows)
                  })
          return tables
import os
# 必须在导入 docling / huggingface 相关库之前设置
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 可选：如果你希望模型缓存到指定磁盘，可以顺便设置
# os.environ["HF_HOME"] = "E:/code/rag_project/models_cache"

from io import BytesIO
from pathlib import Path
from typing import List,Dict,Optional

from docling.document_converter import DocumentConverter,PdfFormatOption
from docling.datamodel.base_models import DocumentStream,InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from .base import BaseParser,ParseResult
from src.parsers.chunker import TextChunker
from src.parsers.cleaner import TextCleaner
from src.parsers.table_parser import TableProcessor


class PDFParser(BaseParser):
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
           
           #初始化文档转换器
           pipeline_options=PdfPipelineOptions()
           pipeline_options.do_ocr=True
           pipeline_options.do_table_structure=True
           
           self.converter=DocumentConverter(
               format_options={
                   InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
               }
           )


    def parse(self,file_path:str) -> ParseResult:
        '''
        解析文档文件
        Args:
            file_path:文件路径
        Returns:
            dict:解析结果
           '''
        #使用docling解析文档
        doc = self.converter.convert(file_path)

        #提取内容
        raw_text=doc.document.export_to_markdown()

        #提取表格
        tables =self._extract_tables(doc)

        #处理数据流程
        cleaned_text=self.cleaner.clean(raw_text)
        marged_tables=self.table_processor.merge_tables(tables)
        chunkes=self.chunker.split(cleaned_text,marged_tables)


        return ParseResult(
             file_name=Path(file_path).name,
             total_pages=len(doc.pages),
             total_chunks=len(chunkes),
             table_count=len(marged_tables),
             chunks=chunkes,
             raw_text=raw_text,
             tables=marged_tables,
             metadata={'parse':'docling','file_path':file_path}
        )
    
    def parse_from_bytes(self,file_bytes:bytes,file_name:str) -> ParseResult:
        '''
        从字节流解析文档
        Args:
            file_bytes:文档字节流
            file_name:文档名
        Returns:解析结果
        '''
        file_name=Path(file_name).name

        #使用docling转换成document文档对象
        #docling 2.x 没有 convert_from_bytes，字节流需包装成 DocumentStream
        source=DocumentStream(name=file_name,stream=BytesIO(file_bytes))
        doc=self.converter.convert(source)
        #提取内容
        raw_text=doc.document.export_to_markdown()
        
        #提取表格
        tables =self._extract_tables(doc)
        
        #处理数据流程
        cleaned_text=self.cleaner.clean(raw_text)
        marged_tables=self.table_processor.merge_tables(tables)
        chunkes=self.chunker.split(cleaned_text,marged_tables)
        
        
        return ParseResult(
            file_name=file_name,
            total_pages=len(doc.pages),
            total_chunks=len(chunkes),
            table_count=len(marged_tables),
            chunks=chunkes,
            raw_text=raw_text,
            tables=marged_tables,
            metadata={'parse':'docling','file_path':file_name}
        )

    def supported_extensions(self) -> list[str]:
        '''
        返回支持的文件扩展名
        Returns:
            支持的文件扩展列名,如:['.pdf','.PDF']
        '''
        return ['.pdf','.PDF']

    def _extract_tables(self,doc):
        '''
        提取表格
        '''
        tables=[]

        #判断有没有tables
        if not hasattr(doc.document,'tables') or not doc.document.tables:
             return tables

        for i,table in enumerate(doc.document.tables,1):
             try:
                #获取页码
                prov=getattr(table,'prov',[])
                page_no=prov[0].page_no if prov else None

                table_data={
                    'table_number':i,
                    'page':page_no,
                    'markdown':table.export_to_markdown(doc=doc.document)
                }
                tables.append(table_data)
             except Exception as e:
                print(f'警告：无法处理表格：{e}')
                continue
        return tables


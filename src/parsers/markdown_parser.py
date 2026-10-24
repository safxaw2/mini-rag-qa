from typing import Optional
from pathlib import Path

from .base import BaseParser,ParseResult
from .chunker import TextChunker,ChunkStrategy
from .cleaner import TextCleaner

class MarkdownParser(BaseParser):
    '''
    markdown解析器
    '''

    def __init__(
            self,
            chunker:Optional[TextChunker]=None,
            cleaner:Optional[TextCleaner]=None,
    ):
        self.cleaner=cleaner if cleaner else TextCleaner()
        self.chunker=chunker if chunker  else TextChunker(
            strategy=ChunkStrategy.PARAGRAPH,
            max_chunk_size=1000,
            chunk_overlap=200
        )

    def parse(self,file_path:str) -> ParseResult:
        '''
        解析markdown文件
        '''
        with open(file_path,'r',encoding='utf-8') as f:
            raw_text=f.read()

        cleaned_text=self.cleaner.clean(raw_text)
        chunks=self.chunker.split(cleaned_text)

        return ParseResult(
            file_name=Path(file_path).name,
            total_pages=1,
            total_chunks=len(chunks),
            table_count=0,
            chunks=chunks,
            raw_text=raw_text,
            tables=[],
            metadata={'parser':'markdown','file_name':Path(file_path).name}
        )

    def parse_from_bytes(self,file_bytes:bytes,file_name:str):
        raw_text=file_bytes.decode('utf-8')
        cleaned_text=self.cleaner.clean(raw_text)
        chunks=self.chunker.split(cleaned_text)

        return ParseResult(
            file_name=Path(file_name).name,
            total_pages=1,
            total_chunks=len(chunks),
            table_count=0,
            chunks=chunks,
            raw_text=raw_text,
            tables=[],
            metadata={'parser':'markdown','file_name':file_name}
        )        
        return

    def supported_extensions(self) -> list[str]:
        return ['.md','.MD','.markdown','.MARKDOWN']


from pathlib import Path
from typing import Optional
from bs4 import BeautifulSoup
from markdownify import markdownify

from .base import BaseParser, ParseResult
from .cleaner import TextCleaner
from .chunker import TextChunker, ChunkStrategy

class HTMLParser(BaseParser):
  """HTML文档解析器"""

  def __init__(
    self,
    chunker: Optional[TextChunker] = None,
    cleaner: Optional[TextCleaner] = None
   ):
    self.cleaner = cleaner or TextCleaner()
    self.chunker = chunker or TextChunker(
      strategy=ChunkStrategy.PARAGRAPH,
      max_chunk_size=1000,
      chunk_overlap=200
     )

  def parse(self, file_path: str) -> ParseResult:
    """解析HTML文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
      html = f.read()

    # 转换为Markdown
    raw_text = markdownify(html)

    cleaned_text = self.cleaner.clean(raw_text)
    chunks = self.chunker.split(cleaned_text)

    return ParseResult(
      file_name=Path(file_path).name,
      total_pages=1,
      total_chunks=len(chunks),
      table_count=0,
      chunks=chunks,
      raw_text=cleaned_text,
      tables=[],
      metadata={"parser": "markdownify"}
     )

  def parse_from_bytes(self, file_bytes: bytes, filename: str) -> ParseResult:
    """从字节流解析HTML"""
    html = file_bytes.decode('utf-8')
    raw_text = markdownify(html)

    cleaned_text = self.cleaner.clean(raw_text)
    chunks = self.chunker.split(cleaned_text)
    return ParseResult(
      file_name=Path(filename).name,
      total_pages=1,
      total_chunks=len(chunks),
      table_count=0,
      chunks=chunks,
      raw_text=cleaned_text,
      tables=[],
      metadata={"parser": "markdownify"}
     )
  
  def supported_extensions(self) -> list:
    return [".html", ".htm", ".HTML", ".HTM"]

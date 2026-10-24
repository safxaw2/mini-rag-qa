from pathlib import Path

from .base import BaseParser,ParseResult
from .pdf_parse import PDFParser
from .html_parser import HTMLParser
from .markdown_parser import MarkdownParser
from .word_paser import Wordarser

class ParserFactory:
    '''
    解析器工厂
    '''
    _parsers={
        ".pdf":PDFParser,
        ".html":HTMLParser,
        ".htm":HTMLParser,
        ".md":MarkdownParser,
        ".markdown":MarkdownParser,
        ".docx":Wordarser
    }

    @classmethod
    def get_parser(cls,file_path:str) -> BaseParser:
        '''
        根据文件拓展名获取解析器
        Args:
            file_path (str): 文件路径
            Returns:        解析器实例
        '''
        ext=Path(file_path).suffix.lower()

        #判断格式
        if ext not in cls._parsers:
            raise ValueError(f"不支持的文件格式:{ext}")

        #获取解析器：注册的既可以是解析器类，也可以是解析器实例
        parser=cls._parsers[ext]
        return parser() if isinstance(parser,type) else parser

    @classmethod
    def register_parser(cls,ext:str,parser):
        '''
        注册解析器
        Args:
            ext (str): 文件拓展名，可带或不带前导点，大小写不敏感
            parser (BaseParser | type[BaseParser]): 解析器类或实例
        '''
        ext=ext.lower()
        if not ext.startswith('.'):
            ext='.'+ext
        cls._parsers[ext]=parser

    @classmethod
    def supported_extensions(cls) -> list[str]:
        '''
        获取支持的格式
        Returns: 支持的格式列表
        '''
        return list(cls._parsers.keys())

def parse_file(file_path:str) -> ParseResult:
    '''
    自动识别文件类型解析
    Args:
        file_path (str): 文件路径
    Returns: 解析结果
    '''
    parser=ParserFactory.get_parser(file_path)
    return parser.parse(file_path)
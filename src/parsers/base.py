#抽象类
from abc import ABC,abstractmethod
from typing import List,Dict
from dataclasses import dataclass

@dataclass
class ParseResult:
    '''解析结果统一格式'''
    file_name: str          # 文件名
    total_pages: int       # 总页数
    total_chunks: int      # 文本块数
    table_count: int       # 表格数
    chunks: List[Dict]     # 文本块列表
    raw_text: str          # 原始文本
    tables: List[Dict]     # 表格列表
    metadata: Dict         # 其他元数据

class BaseParser(ABC):
    '''
    文件解析基类
    '''

    @abstractmethod
    def parse(self,file_path:str) -> dict:
        '''
        解析文档文件
        Args:
            file_path:文件路径
        Returns:
            dict:解析结果
        '''
        pass

    @abstractmethod
    def parse_from_bytes(self,file_bytes:bytes,file_name:str) -> dict:
        '''
        从字节流解析文档
        Args:
            file_bytes:文档字节流
            file_name:文档名
        Returns:解析结果
        '''
        pass

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        '''
        返回支持的文件扩展名
        Returns:
            支持的文件扩展列名,如:['.pdf','.PDF']
        '''
        pass
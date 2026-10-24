from abc import ABC,abstractmethod
from typing import List,Dict

class BaseEmbedding(ABC):
    '''
    嵌入模型抽象类
    '''
    @abstractmethod
    def embed(self,text:str) -> List[float]:
        '''
        嵌入单个文本
        '''
        pass

    @abstractmethod
    def embed_batch(self,texts:List[str]) -> List[List[float]]:
        '''
        嵌入多个文本
        '''
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        '''
        获取嵌入维度
        '''
        pass

    @abstractmethod
    def get_name(self) -> str:
        '''
        获取模型名称
        '''
        pass
from typing import List

from openai import OpenAI
from tenacity import retry,stop_after_attempt,wait_exponential

from src.embeddings.base import BaseEmbedding

class OpenAIEmbedding(BaseEmbedding):

    def __init__(
            self,
            api_key:str,
            model:str,
            base_url:str,
            dimension:int=None
        ):
        self.api_key=api_key
        self.model=model
        self.base_url=base_url
        self.dimension=dimension

        self.client=OpenAI(api_key=api_key,base_url=base_url)

    def embed_text(self, text:str) -> List[float]:
        '''
        处理单个文本
        '''
        rs=self.embed_batch([text])
        return rs[0]

    def embed(self, text:str) -> List[float]:
        '''
        实现 BaseEmbedding.embed 抽象方法，等价于 embed_text
        '''
        return self.embed_text(text)

    @retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1,min=2,max=10))#重试3次 第一次延迟2秒，第二次延迟4秒，第三次延迟8秒
    def embed_batch(self, texts:List[str]) -> List[List[float]]:
        '''
        批量处理文本
        '''
        params={
            "input":texts,
            "model":self.model
        }
        #判断是否指定维度
        if self.dimension:
            params["dimensions"]=self.dimension
        #发送请求
        resp=self.client.embeddings.create(**params)
        #提取向量
        embeddings=[item.embedding for item in resp.data]
        return embeddings

    def get_dimension(self) -> int:
        '''
        获取嵌入维度
        '''
        return self.dimension

    def get_name(self) -> str:
        '''
        获取模型名称
        '''
        return self.model

from sentence_transformers import SentenceTransformer
from typing import List

from src.embeddings.base import BaseEmbedding

class LocalEmbedding(BaseEmbedding):
    def __init__(self, model_name:str,device:str='cuda'):
        '''
        初始化本地嵌入模型
     Args:
       model_name: 模型名称（HuggingFace）
         - BAAI/bge-small-zh-v1.5: 中文推荐（默认，性能优秀）
         - BAAI/bge-base-zh-v1.5: 中文高质量（更大，效果更好）
         - BAAI/bge-large-zh-v1.5: 中文顶级效果（最大，资源消耗高）
         - shibing624/text2vec-base-chinese: 中文通用（经典模型）
         - moka-ai/m3e-base: 中文高质量
         - sentence-transformers/all-MiniLM-L6-v2: 英文
            device: 运行设备（cpu / cuda）
        '''
        self.model_name = model_name
        self.device = device

        # 加载模型
        self.model = SentenceTransformer(model_name,device=device)

    def embed_text(self, text:str) -> List[float]:
        '''获取文本的向量'''
        embedding=self.model.encode(text,convert_to_tensor=True)
        return embedding.tolist()

    def embed(self, text:str) -> List[float]:
        '''实现 BaseEmbedding.embed 抽象方法，等价于 embed_text'''
        return self.embed_text(text)

    def embed_batch(self, texts:List[str]) -> List[List[float]]:
        '''
        批量嵌入文本
        '''
        embeddings=self.model.encode(texts,convert_to_tensor=True)
        return embeddings.tolist()

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    def get_name(self) -> str:
        return self.model_name
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class Document:
  """文档对象"""
  id: str                # 文档ID
  content: str              # 文档内容
  metadata: Dict[str, Any]        # 元数据
  embedding: Optional[List[float]] = None # 向量（可选）

@dataclass
class SearchResult:
  """检索结果"""
  document: Document           # 文档
  score: float              # 相似度分数
  metadata: Dict[str, Any]        # 额外元数据

class BaseVectorStore(ABC):
  @abstractmethod
  def add_documents(self,
                    documents: List[Document],
                    embeddings: List[List[float]],
                    ) -> List[str ]:
    '''
    添加文档到数据库
    参数：
      documents: 文档列表
      embeddings: 文档对应的嵌入向量列表
      返回：
      文档ID列表
    '''
    pass

  @abstractmethod
  def search(self, 
             query_embedding:List[float],
             top_k: int = 5,
             filter:Dict[str, Any] = None,
             ) -> List[SearchResult]:
    '''
    搜索文档
    参数：
      query: 查询字符串
      k: 返回的文档数量
      返回：
      搜索结果列表
    '''
    pass

  @abstractmethod
  def get_collection_stats(self) -> Dict[str, int]:
    '''
    获取集合名称
    返回：
    统计信息（文档数量、维度）
    '''
    pass

  @abstractmethod
  def clear(self) -> bool:
    '''
    清理数据库
    返回：
    清理成功与否
    '''
    pass
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
             filter:Optional[Dict[str, Any]] = None,
             ) -> List[SearchResult]:
    '''
    按向量检索文档
    ARGS:
      query_embedding: 查询向量
      top_k: 返回结果数上限
      filter: 元数据过滤条件（Chroma where 语法），None 表示不过滤
    RETURN:
      按距离升序的 SearchResult 列表；score 为距离（cosine 下越小越相似）
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
import time
import copy
import json
import threading
from collections import OrderedDict

from typing import Dict, Optional, Any,List
from dataclasses import dataclass

from src.database.document_store import DocumentStore


@dataclass
class RetrievalConfig:
  """检索配置。

  Attributes:
    top_k: 最多返回的结果数。
    max_distance: 余弦距离上限，只保留 distance <= max_distance 的结果；
      距离越小越相似。None 表示不按距离过滤。
    namespace: 命名空间过滤；None 表示不限制。
    metadata_filter: 传给向量库的元数据过滤条件，如 {"category": "fruit"}；
      None 表示不过滤。
  """
  top_k: int = 5            # 返回结果数
  max_distance: Optional[float] = None  # 距离上限（越小越相似）
  namespace: Optional[str] = None    # 命名空间过滤
  metadata_filter: Optional[Dict] = None # 元数据过滤


@dataclass
class SearchResponse:
  """检索结果。

  Attributes:
    query: 原始查询字符串。
    total_results: results 的长度（已按距离过滤并截断 top_k 之后）。
    results: 每项为 {"id": str, "content": str, "metadata": dict,
      "score": float}；score 是余弦距离，越小越相似。
    retrieval_time_ms: 本次检索耗时（毫秒，含向量化与向量库查询）。
  """
  query: str
  total_results: int
  results:List[Dict[str,Any]]
  retrieval_time_ms: int


class SearchService:
  """检索服务 - 统一的检索入口"""

  def __init__(
      self,
      document_store: DocumentStore,
      cache_ttl_seconds: float = 30.0,
      cache_max_size: int = 256,
  ):
    """初始化检索服务。

    Args:
      document_store: 文档存储实例。
    """
    self.store = document_store
    self.cache_ttl_seconds = max(0.0, cache_ttl_seconds)
    self.cache_max_size = max(0, cache_max_size)
    self._cache = OrderedDict()
    self._cache_lock = threading.Lock()

  def search(self,query:str,config:Optional[RetrievalConfig]=None)->SearchResponse:
    """执行检索。

    Args:
      query: 查询内容。
      config: 检索配置；为 None 时使用 RetrievalConfig() 的默认值。

    Returns:
      SearchResponse。results 每项包含 id、content、metadata、
      score（余弦距离，越小越相似）。

    Raises:
      Exception: 嵌入模型或向量库的异常会原样向上抛出。
    """
    #开始时间
    start_time=time.perf_counter()

    #判断config是否传递
    if config is None:
      config=RetrievalConfig()

    cache_key=(
      query,
      config.top_k,
      config.max_distance,
      config.namespace,
      json.dumps(config.metadata_filter,sort_keys=True,separators=(",",":"),default=str),
      getattr(self.store,"revision",0),
    )
    if self.cache_ttl_seconds and self.cache_max_size:
      now=time.monotonic()
      with self._cache_lock:
        cached=self._cache.get(cache_key)
        if cached is not None:
          expires_at,response=cached
          if expires_at > now:
            self._cache.move_to_end(cache_key)
            end_time=time.perf_counter()
            return SearchResponse(
              query=response.query,
              total_results=response.total_results,
              results=copy.deepcopy(response.results),
              retrieval_time_ms=round((end_time-start_time)*1000),
            )
          del self._cache[cache_key]

    #交给 DocumentStore 完成向量化与过滤（namespace/metadata_filter 已合并为 $and）
    results=self.store.search(
      query=query,
      top_k=config.top_k,
      namespace=config.namespace,
      metadata_filter=config.metadata_filter,
    )

    #数据过滤(score 是余弦距离，越小越相似)
    filtered_results=[
      result for result in results
      if config.max_distance is None or result["score"]<=config.max_distance
    ]
    #top_k
    results=filtered_results[:config.top_k]

    #结束时间
    end_time=time.perf_counter()

    response=SearchResponse(
      query=query,
      total_results=len(results),
      results=results,
      retrieval_time_ms=round((end_time-start_time)*1000)
    )
    if self.cache_ttl_seconds and self.cache_max_size:
      with self._cache_lock:
        self._cache[cache_key]=(
          time.monotonic()+self.cache_ttl_seconds,
          copy.deepcopy(response),
        )
        self._cache.move_to_end(cache_key)
        while len(self._cache)>self.cache_max_size:
          self._cache.popitem(last=False)
    return response

  def get_stats(self)->Dict[str,Any]:
    """获取检索服务统计信息。

    Returns:
      count (int)、name (str)、distance (str)、
      embedding_dim (int)、embedding_model (str)。
    """
    return self.store.get_stats()

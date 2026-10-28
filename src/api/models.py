from pydantic import BaseModel
from pydantic import Field
from typing import List,Dict,Any,Optional

class ChatRequest(BaseModel):
    query: str=Field(...,description="用户问题")
    collection_name: str=Field(...,description="数据集名称")
    model:Optional[str]=Field(None,description="LLM模型名称")

class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    tokens_used: int
    total_time_ms: int
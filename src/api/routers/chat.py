from fastapi import APIRouter
from fastapi.responses import StreamingResponse
import json
import logging
import time
from src.api.models import ChatRequest,ChatResponse
from src.api.config import settings
from src.api.deps import get_search_service,get_llm
from src.retrieval.search_service import RetrievalConfig
from src.llm.rag import generate_answer,generate_answer_stream


router=APIRouter(prefix='/chat',tags=['chat'])
logger=logging.getLogger(__name__)

@router.post('/',response_model=ChatResponse)
def chat(request:ChatRequest):
    '''
    RAG问答接口
    '''
    #开始计时
    start_time=time.time()
    #确定模型是否存在
    model=request.model or settings.DEFAULT_MODEL
    #获取模型
    llm=get_llm(model)

    #获取搜索服务
    search_service=get_search_service(request.collection_name)
    #创建检索配置
    search_config=RetrievalConfig(top_k=settings.SEARCH_TOP_K)
    #检索
    search_response=search_service.search(request.query,search_config)

    #生成答案
    result=generate_answer(
        llm=llm,
        question=request.query,
        result=search_response.results,
        temperature=settings.LLM_TEMPERATURE,
        max_token=settings.LLM_MAX_TOKENS
    )

    #构建引用来源
    sources=[
        {
            'id':r.get('id'),
            'score':r.get('score'),
            'text':r.get('content', '')[:200] + '...' if len(r.get('content', '')) > 200 else r.get('content', ''),
        }
        for r in search_response.results[:3]
    ]

    #计算总时间
    total_time_ms=int((time.time()-start_time)*1000)
    return ChatResponse(
        query=request.query,
        answer=result['answer'],
        sources=sources,
        tokens_used=result.get('tokens_used', {}).get('total', 0),
        total_time_ms=total_time_ms
    )


@router.post('/stream')
def chat_stream(request:ChatRequest):
    model=request.model or settings.DEFAULT_MODEL
    llm=get_llm(model)
    search_service=get_search_service(request.collection_name)
    search_response=search_service.search(
        request.query,
        RetrievalConfig(top_k=settings.SEARCH_TOP_K),
    )
    sources=[
        {
            'id':result.get('id'),
            'score':result.get('score'),
            'text':(
                result.get('content','')[:200] + '...'
                if len(result.get('content','')) > 200
                else result.get('content','')
            ),
        }
        for result in search_response.results[:3]
    ]

    def events():
        try:
            for content in generate_answer_stream(
                llm=llm,
                question=request.query,
                result=search_response.results,
                temperature=settings.LLM_TEMPERATURE,
                max_token=settings.LLM_MAX_TOKENS,
            ):
                yield f"data: {json.dumps({'type':'token','content':content},ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type':'done','query':request.query,'sources':sources},ensure_ascii=False)}\n\n"
        except Exception:
            logger.exception("Streaming chat generation failed")
            yield f"data: {json.dumps({'type':'error','message':'流式生成失败'},ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type='text/event-stream',
        headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'},
    )
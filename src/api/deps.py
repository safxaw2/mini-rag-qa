#接口依赖
from functools import lru_cache
import threading

from fastapi import HTTPException

from src.database.document_store import DocumentStore
from src.retrieval.search_service import SearchService
from src.embeddings.local_embedding import LocalEmbedding
from src.embeddings.openai_embedding import OpenAIEmbedding
from src.embeddings.chroma_store import ChromaVectorStore
from src.api.config import settings
from src.llm.llm import LLM

_resource_cache_lock=threading.RLock()


@lru_cache(maxsize=8)
def _get_embedding_model(provider, model_name, api_key, base_url, dimension, device):
    if provider == 'local':
        return LocalEmbedding(model_name=model_name, device=device)
    if provider == 'openai':
        if not api_key:
            raise ValueError("EMBEDDING_API_KEY or DASHSCOPE_API_KEY is required for online embeddings")
        return OpenAIEmbedding(
            api_key=api_key,
            model=model_name,
            base_url=base_url,
            dimension=dimension,
        )
    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


def get_embedding_model():
    provider=settings.EMBEDDING_PROVIDER
    model_name=(
        settings.LOCAL_EMBEDDING_MODEL
        if provider == 'local'
        else settings.EMBEDDING_MODEL
    )
    with _resource_cache_lock:
        return _get_embedding_model(
            provider,
            model_name,
            settings.EMBEDDING_API_KEY,
            settings.EMBEDDING_BASE_URL,
            settings.EMBEDDING_DIMENSION,
            settings.LOCAL_EMBEDDING_DEVICE,
        )


@lru_cache(maxsize=64)
def _get_document_store(collection_name:str):
    vector_store=ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=settings.VECTOR_STORE_DIRECTORY,
        distance=settings.VECTOR_DISTANCE,
        search_ef=settings.VECTOR_HNSW_SEARCH_EF,
        construction_ef=settings.VECTOR_HNSW_CONSTRUCTION_EF,
        hnsw_m=settings.VECTOR_HNSW_M,
    )
    return DocumentStore(
        embedding_model=get_embedding_model(),
        collection_name=collection_name,
        vector_store=vector_store,
    )


def get_document_store(collection_name:str):
    with _resource_cache_lock:
        return _get_document_store(collection_name)


@lru_cache(maxsize=64)
def _get_search_service(collection_name:str):
    return SearchService(
        document_store=get_document_store(collection_name),
        cache_ttl_seconds=settings.RETRIEVAL_CACHE_TTL_SECONDS,
        cache_max_size=settings.RETRIEVAL_CACHE_MAX_SIZE,
    )


def get_search_service(collection_name:str):
    with _resource_cache_lock:
        return _get_search_service(collection_name)


def get_llm(model:str):
    '''
    
    获取llm'''
    config = settings.LLM_PROVIDER.get(model)
    if config is None:
        config = next(
            (
                provider_config
                for provider_config in settings.LLM_PROVIDER.values()
                if provider_config.get('default_model') == model
            ),
            None,
        )
    if config is None:
        supported_models = {
            model_name
            for provider_name, provider_config in settings.LLM_PROVIDER.items()
            for model_name in (provider_name, provider_config.get('default_model'))
            if model_name
        }
        raise HTTPException(
            status_code=400,
            detail=f"模型{model}不支持,目前支持的模型有：{sorted(supported_models)}",
        )

    with _resource_cache_lock:
        return _get_llm(
            config.get('default_model'),
            config.get('api_key'),
            config.get('base_url'),
        )


@lru_cache(maxsize=32)
def _get_llm(model:str, api_key:str, base_url:str):
    return LLM(model=model,api_key=api_key,base_url=base_url)
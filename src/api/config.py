import os

from dotenv import load_dotenv

load_dotenv()


def _int_setting(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _float_setting(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


class Settings:
    '''
    api settings
    '''
    #文件上传配置
    MAX_FILE_SIZE = 1024 * 1024 * 50    #最大文件大小50M
    ALLOWED_EXTENSIONS = ['.pdf','.docx','html','.md']
    #上传文件存储目录
    UPLOAD_DIR='data/uploads'

    # Embedding model configuration. Local embeddings remain the default.
    EMBEDDING_PROVIDER = os.getenv('EMBEDDING_PROVIDER', 'local').lower()
    LOCAL_EMBEDDING_MODEL = os.getenv('LOCAL_EMBEDDING_MODEL', 'BAAI/bge-small-zh-v1.5')
    LOCAL_EMBEDDING_DEVICE = os.getenv('LOCAL_EMBEDDING_DEVICE', 'cpu')
    EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-v3')
    EMBEDDING_API_KEY = os.getenv('EMBEDDING_API_KEY') or os.getenv('DASHSCOPE_API_KEY')
    EMBEDDING_BASE_URL = os.getenv(
        'EMBEDDING_BASE_URL',
        'https://dashscope.aliyuncs.com/compatible-mode/v1',
    )
    EMBEDDING_DIMENSION = _int_setting('EMBEDDING_DIMENSION', 0) or None

    # Vector index configuration (takes effect when a collection is created).
    VECTOR_STORE_DIRECTORY = os.getenv('VECTOR_STORE_DIRECTORY', './data/chroma_db')
    VECTOR_DISTANCE = os.getenv('VECTOR_DISTANCE', 'cosine')
    VECTOR_HNSW_SEARCH_EF = _int_setting('VECTOR_HNSW_SEARCH_EF', 100)
    VECTOR_HNSW_CONSTRUCTION_EF = _int_setting('VECTOR_HNSW_CONSTRUCTION_EF', 100)
    VECTOR_HNSW_M = _int_setting('VECTOR_HNSW_M', 16)

    #LLM配置
    LLM_PROVIDER={
        'qwen':{
            'api_key':os.getenv('DASHSCOPE_API_KEY'),
            'base_url':os.getenv('LLM_BASE_URL',os.getenv('EMBEDDING_BASE_URL','https://api.dashscope.com')),
            'default_model':os.getenv('LLM_MODEL','qwen3.7-flash'),
        }
    }

    #默认模型
    DEFAULT_MODEL = os.getenv('DEFAULT_MODEL',os.getenv('LLM_MODEL','qwen3.7-flash'))

    #LLM生成参数
    LLM_TEMPERATURE = 0.7
    LLM_MAX_TOKENS = 2000

    #检索配置
    SEARCH_TOP_K = _int_setting('RETRIEVAL_TOP_K', 5)
    SERACH_TOP_K = SEARCH_TOP_K
    RETRIEVAL_CACHE_TTL_SECONDS = _float_setting('RETRIEVAL_CACHE_TTL_SECONDS', 30.0)
    RETRIEVAL_CACHE_MAX_SIZE = _int_setting('RETRIEVAL_CACHE_MAX_SIZE', 256)

#全局配置对象
settings = Settings()
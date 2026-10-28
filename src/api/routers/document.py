import os
import hashlib
from pathlib import Path
from typing import Optional
from chromadb.errors import InvalidArgumentError
from fastapi import APIRouter,UploadFile,File,HTTPException
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from ..config import settings
from src.parsers import ParserFactory,parse_file
from src.embeddings.chroma_store import ChromaVectorStore
from src.api.deps import get_document_store
router = APIRouter(prefix='/api', tags=['文件解析'])

class CollectionRequest(BaseModel):
    collection_name: str

@router.get('/collections')
async def list_collections():
    '''列出现有知识库。'''
    return {"collections":ChromaVectorStore.list_collections(settings.VECTOR_STORE_DIRECTORY)}

@router.post('/collections')
async def create_collection(body:CollectionRequest):
    '''创建知识库（已存在时保持不变）。'''
    collection_name=body.collection_name.strip()
    if not collection_name:
        raise HTTPException(status_code=400,detail="知识库名称不能为空")
    try:
        ChromaVectorStore(
            collection_name=collection_name,
            persist_directory=settings.VECTOR_STORE_DIRECTORY,
            distance=settings.VECTOR_DISTANCE,
            search_ef=settings.VECTOR_HNSW_SEARCH_EF,
            construction_ef=settings.VECTOR_HNSW_CONSTRUCTION_EF,
            hnsw_m=settings.VECTOR_HNSW_M,
        )
    except InvalidArgumentError as e:
        raise HTTPException(status_code=400,detail=f'知识库名称无效或无法创建：{str(e)}')
    return {"success":True,"collection_name":collection_name}

@router.post('/upload')
async def upload_document(file: UploadFile = File(...),collection_name:Optional[str]=None):
    '''
    上传文件解析文件
    支持的文件类型：pdf,docx,html,md
    '''

    ext=os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ParserFactory.supported_extensions():
        raise HTTPException(status_code=400,detail=f'不支持的文件类型{ext}')

    file_bytes=await file.read(settings.MAX_FILE_SIZE+1)

    #判断文件大小
    if len(file_bytes)>settings.MAX_FILE_SIZE:
        size_mb=settings.MAX_FILE_SIZE/1024/1024
        raise HTTPException(status_code=400,detail=f'文件过大。最大允许{size_mb}MB') #10M

    try:
        file_hash=hashlib.sha256(file_bytes).hexdigest()
        store=None
        if collection_name:
            store=await run_in_threadpool(get_document_store,collection_name)
            existing_chunks=await run_in_threadpool(
                store.count_file_hash,
                file_hash,
                collection_name,
            )
            if existing_chunks:
                return {
                    'success':True,
                    'file_name':file.filename,
                    'stored':True,
                    'duplicate':True,
                    'doc_count':existing_chunks,
                }

        #获取解析器
        parser=ParserFactory.get_parser(file.filename)
        #解析文件
        rs=await run_in_threadpool(parser.parse_from_bytes,file_bytes,file.filename)
        if getattr(rs,'metadata',None) is None:
            rs.metadata={}
        rs.metadata['file_hash']=file_hash
        #保存到向量数据库
        if store is not None:
            doc_ids=await run_in_threadpool(
                store.add_parse_result,
                rs,
                name_space=collection_name,
            )
        else:
            doc_ids=[]

        #Persist each successfully parsed upload only once for this request.
        upload_dir=_ensure_upload_dir()
        file_name=f'{file_hash}{ext}'
        file_path=upload_dir/file_name
        if not file_path.exists():
            await run_in_threadpool(file_path.write_bytes,file_bytes)

        return {
            'success':True,
            'file_name':rs.file_name,
            'stored':store is not None,
            'doc_count':len(doc_ids) if store is not None else rs.total_chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'文件解析失败：{str(e)}')

from pydantic import BaseModel
class ParsePathRequest(BaseModel):
    file_path: str

def _ensure_upload_dir():
    '''
    确保上传目录存在
    '''
    upload_dir=Path(settings.UPLOAD_DIR)    # 上传目录
    upload_dir.mkdir(parents=True,exist_ok=True)    # 创建目录
    return upload_dir

@router.post('/parse-path')
async def parse_document_by_path(body:ParsePathRequest):
    '''
    按服务器路径解析文件（用于已有文档或批处理）
    '''
    path=body.file_path
    #判断path是不是完整路径
    if os.path.dirname(path):
        traget_path=path
    else:
        #传递文件名称
        upload_dir=_ensure_upload_dir()
        traget_path=str(upload_dir/path)

    #判断文件是否存在
    if not os.path.isfile(traget_path):
        raise HTTPException(status_code=400,detail=f'文件不存在：{traget_path}')

    try:
        rs=parse_file(traget_path)
        return {
            'msg':'处理成功',
            'file_name':rs.file_name,
            'total_pages':rs.total_pages,
            'total_chunks':rs.total_chunks,
            'table_count':rs.table_count,
            'chunks':rs.chunks,
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=f'文件解析失败：{str(e)}')

@router.get('/supported-extensions')
async def supported_extensions():
    '''
    获取支持的文件类型
    '''
    return {
        'formats':ParserFactory.supported_extensions()
    }    
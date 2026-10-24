import os
from pathlib import Path
import time
from fastapi import APIRouter,UploadFile,File,HTTPException

from ..config import settings
from src.parsers import ParserFactory,parse_file

router = APIRouter(prefix='/api', tags=['文件解析'])

def _genreate_file_name(file_name:str) -> str:
    '''
    生成文件名
    '''
    name,ext=os.path.splitext(file_name)
    timestamp=time.time()
    return f'{name}_{timestamp}{ext}'

@router.post('/upload')
async def upload_document(file: UploadFile = File(...)):
    '''
    上传文件解析文件
    支持的文件类型：pdf,docx,html,md
    '''

    ext=os.path.splitext(file.filename or "")[-1].lower()
    file_bytes=await file.read()

    #判断文件大小
    if len(file_bytes)>settings.MAX_FILE_SIZE:
        size_mb=settings.MAX_FILE_SIZE/1024/1024
        raise HTTPException(status_code=400,detail=f'文件过大。最大允许{size_mb}MB') #10M
    #判断文件类型
    if ext not in ParserFactory.supported_extensions(): #判断文件类型
        raise HTTPException(status_code=400,detail=f'不支持的文件类型{ext}')

    #数据保存
    upload_dir=_ensure_upload_dir()
    #文件名：应对文件名重复,对上传的文件统一管理
    file_name=_genreate_file_name(file.filename)
    file_path=str(upload_dir/file_name)
    with open(file_path,'wb') as f:
        f.write(file_bytes)

    try:
        #获取解析器
        parser=ParserFactory.get_parser(file.filename)
        #解析文件
        rs=parser.parse_from_bytes(file_bytes,file.filename)
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
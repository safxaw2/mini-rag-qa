from fastapi import APIRouter
import time

router = APIRouter(prefix="/health", tags=["健康检查"])


@router.get("/")
async def health():
  """健康检查"""
  return {"status": "healthy", "timestamp": int(time.time())}

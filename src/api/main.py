import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routers import document
from src.api.routers import chat
from src.api.routers import health

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontent"

app=FastAPI(
    title='FastAPI RAG系统',
    description='基于FastAPI的RAG系统',
    version='0.1'
)

app.mount("/frontent", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
app.include_router(document.router)
app.include_router(chat.router,prefix='/api')
app.include_router(health.router,prefix='/api')
@app.get('/')
async def index():
    return FileResponse(FRONTEND_DIR / "index.html")

if __name__=='__main__':
    import uvicorn
    uvicorn.run('src.api.main:app',host='localhost',port=8000,reload=True)

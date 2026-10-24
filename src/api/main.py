import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI

from src.api.routers import document

app=FastAPI(
    title='FastAPI RAG系统',
    description='基于FastAPI的RAG系统',
    version='0.1'
)

app.include_router(document.router)
@app.get('/')
async def index():
    return 'Hello World'

if __name__=='__main__':
    import uvicorn
    uvicorn.run('src.api.main:app',host='localhost',port=8000,reload=True)
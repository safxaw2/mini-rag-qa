
import uuid
from typing import List, Dict,Any

import chromadb

from src.embeddings.vector_store import BaseVectorStore,Document,SearchResult

class ChromaVectorStore(BaseVectorStore):
    '''
    向量数据存储实现
    '''
    def __init__(
            self,
            collection_name:str='rag_documents',
            persist_directory:str='./data/chroma_db',
            distance:str='cosine'
    ):
        self.collection_name=collection_name
        self.distance=distance
        #创建客户端
        self.client=chromadb.PersistentClient(path=persist_directory)

        #获取或创建集合
        self.collection=self.client.get_or_create_collection(name=collection_name,metadata={'hnsw:space':distance})

    def add_documents(
            self,
            documents: List[Document],
            embeddings: List[List[float]],
        ) -> List[str]:
        '''
        添加文档
        '''
        if not documents:
            return []

        ids=[]
        metadata=[]
        texts=[]
        for doc in documents:
            doc_id=doc.id or str(uuid.uuid4())
            ids.append(doc_id)
            metadata.append(doc.metadata)
            texts.append(doc.content)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadata,
            embeddings=embeddings
        )
        return ids

    def search(self, query_embedding, top_k = 5, filter = None):
        results=self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter
        )
        search_results=[]
        if results['ids'] and results['ids'][0]:
            for i,doc_id in enumerate(results['ids'][0]):
                search_results.append(
                    SearchResult(
                        document=Document(
                            id=doc_id,
                            content=results['documents'][0][i],
                            metadata=results['metadatas'][0][i] if results.get('metadatas') else {}
                        ),
                        metadata={},
                        score=results['distances'][0][i] if results.get('distances') else 0,
                    ),
                )
        return search_results

    def clear(self) -> bool:
        '''
        清空所有数据
        '''
        self.client.delete_collection(self.collection_name)
        self.collection=self.client.create_collection(name=self.collection_name,metadata={'hnsw:space':self.distance})
        return True

    def get_collection_stats(self) -> Dict[str,Any]:
        return {
            'count':self.collection.count(),
            'name':self.collection.name,
            'distance':self.distance,
        }
        

import uuid
from functools import lru_cache
import threading
from typing import List, Dict,Any

import chromadb

from src.embeddings.vector_store import BaseVectorStore,Document,SearchResult


_client_cache_lock=threading.Lock()


@lru_cache(maxsize=8)
def _persistent_client(persist_directory: str):
    return chromadb.PersistentClient(path=persist_directory)


def _get_persistent_client(persist_directory: str):
    with _client_cache_lock:
        return _persistent_client(persist_directory)


class ChromaVectorStore(BaseVectorStore):
    '''
    向量数据存储实现
    '''
    def __init__(
            self,
            collection_name:str='rag_documents',
            persist_directory:str='./data/chroma_db',
            distance:str='cosine',
            search_ef:int=100,
            construction_ef:int=100,
            hnsw_m:int=16,
    ):
        self.collection_name=collection_name
        self.distance=distance
        self.persist_directory=persist_directory
        self.collection_metadata={
            'hnsw:space':distance,
            'hnsw:search_ef':search_ef,
            'hnsw:construction_ef':construction_ef,
            'hnsw:M':hnsw_m,
        }
        self.client=_get_persistent_client(persist_directory)

        #获取或创建集合
        self.collection=self.client.get_or_create_collection(name=collection_name,metadata=self.collection_metadata)

    @staticmethod
    def list_collections(persist_directory: str = './data/chroma_db') -> List[str]:
        client=_get_persistent_client(persist_directory)
        return sorted(collection.name for collection in client.list_collections())

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
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")

        ids=[]
        metadata=[]
        texts=[]
        for doc in documents:
            doc_id=doc.id or str(uuid.uuid4())
            ids.append(doc_id)
            metadata.append(doc.metadata)
            texts.append(doc.content)

        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadata,
            embeddings=embeddings
        )
        return ids

    def count_by_metadata(
        self,
        metadata_filter: Dict[str, Any],
        namespace: str = None,
    ) -> int:
        results=self.collection.get(where=metadata_filter,include=['metadatas'])
        if namespace is None:
            return len(results['ids'])
        return sum(
            metadata.get('namespace') == namespace
            for metadata in results['metadatas']
        )

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
        self.collection=self.client.create_collection(name=self.collection_name,metadata=self.collection_metadata)
        return True

    def get_collection_stats(self) -> Dict[str,Any]:
        return {
            'count':self.collection.count(),
            'name':self.collection.name,
            'distance':self.distance,
        }
        
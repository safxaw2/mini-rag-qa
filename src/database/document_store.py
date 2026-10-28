#解析数据—>向量--->向量存储--->搜索-->结果
from typing import Optional,List,Dict,Any
import uuid
from src.embeddings.base import BaseEmbedding
from src.embeddings.chroma_store import ChromaVectorStore
from src.parsers.base import ParseResult
from src.embeddings.vector_store import Document

class DocumentStore:
    def __init__(self,
                 embedding_model:BaseEmbedding,
                 vector_store:Optional[ChromaVectorStore]=None,
                 collection_name:str='rag_documents',
                 ):
        self.embedding=embedding_model
        self.vector_store=vector_store or ChromaVectorStore(collection_name=collection_name)
        self.revision=0

    def add_parse_result(self,parse_result:ParseResult,name_space:str='default')->List[str]:
        #向量存储
        documents=[]
        texts_to_embed=[]
        file_hash=parse_result.metadata.get('file_hash') if parse_result.metadata else None

        #遍历文件
        for chunk in parse_result.chunks:
            content=chunk['content']
            doc_id=(
                f'{name_space}_{file_hash}_{chunk["index"]}'
                if file_hash
                else f'{name_space}_{parse_result.file_name}_{chunk["index"]}'
            )
            metadata={
                "file_name":parse_result.file_name,
                "chunk_index":chunk["index"],
                "chunk_type":chunk.get("type","text"),
                "namespace":name_space,
                "length":chunk.get('length',len(content))
            }
            if file_hash:
                metadata["file_hash"]=file_hash

            #若果是表格，添加页面
            if chunk.get("type")=="table" and chunk.get("page") is not None:
                metadata["page"]=chunk["page"]

            documents.append(
                Document(id=doc_id,content=content,metadata=metadata)
            )
            #需要转成向量内容
            texts_to_embed.append(content)
        if not documents:
            return []
        embeddings=self.embedding.embed_batch(texts_to_embed)
        #add to vector store
        doc_ids=self.vector_store.add_documents(documents,embeddings)
        self.revision+=1
        return doc_ids

    def count_file_hash(self,file_hash:str,name_space:str)->int:
        return self.vector_store.count_by_metadata(
            {"file_hash":file_hash},
            namespace=name_space,
        )

    def search(self,
               query:str,
               top_k:int=5,
               namespace:Optional[str]=None,
               metadata_filter:Optional[Dict[str,Any]]=None) -> List[Dict[str,Any]]:
        '''
        搜索
        ARGS:
            query:查询内容
            top_k:返回数量
            namespace:命名空间；None 表示不限制
            metadata_filter:元数据过滤条件（Chroma where 语法）；None 表示不过滤
        RETURN:
            List[Dict]，每项为 {"id", "content", "metadata", "score"}；
            score 为向量库返回的距离，越小越相似
        '''
        #生成向量
        query_embedding=self.embedding.embed_text(query)

        #组合过滤条件（多个条件用 $and 连接）
        conditions=[]
        if namespace:
            conditions.append({"namespace":namespace})
        if metadata_filter:
            conditions.append(metadata_filter)
        if len(conditions)>1:
            filter_conditions={"$and":conditions}
        elif conditions:
            filter_conditions=conditions[0]
        else:
            filter_conditions=None

        results=self.vector_store.search(query_embedding,top_k,filter_conditions)
        return [
            {
                'id':rs.document.id,
                'content':rs.document.content,
                'metadata':rs.document.metadata,
                'score':rs.score
            } for rs in results
        ]


    def add_documents(self,texts:List[str],metadata:Optional[List[Dict[str,Any]]]=None,namespace:str='default')->List[str]:
        '''
        直接追加向量文档
        '''
        documents=[]
        metadata=metadata or []
        for i,text in enumerate(texts):
            doc_id=f'{namespace}_{uuid.uuid4().hex[:8]}'
            meta=dict(metadata[i]) if i<len(metadata) else {}
            meta['namespace']=namespace
            documents.append(
                Document(id=doc_id,content=text,metadata=meta)
            )
        embeddings=self.embedding.embed_batch(texts)
        doc_ids=self.vector_store.add_documents(documents,embeddings)
        if doc_ids:
            self.revision+=1
        return doc_ids
            
    def get_stats(self)->Dict[str,Any]:
        '''
        获取统计信息
        '''
        stats=self.vector_store.get_collection_stats()
        stats['embedding_dim']=self.embedding.get_dimension()
        stats['embedding_model']=self.embedding.get_name()
        return stats
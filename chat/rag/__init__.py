# sentence transformers
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from django.conf import settings
from llama_index.vector_stores.postgres import PGVectorStore
from .vector_retriever import VectorDBRetriever


def setup_rag(table_name: str = 'llama2_paper_v3', number_of_results:int=10, max_tokens:int=500) -> PGVectorStore:

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
    db_config = settings.DATABASES['default']
    vector_store = PGVectorStore.from_params(
        database=db_config['NAME'],
        host=db_config['HOST'],
        password=db_config['PASSWORD'],
        port=db_config['PORT'],
        user=db_config['USER'],
        table_name=table_name,  # Replace with your table name
        embed_dim=384,  # Adjust according to your embedding dimension
    )
    return VectorDBRetriever(vector_store=vector_store, 
                             embed_model=embed_model, 
                             query_mode="default",
                             similarity_top_k=number_of_results, 
                             max_tokens=max_tokens)
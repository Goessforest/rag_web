import threading
import os, json
from django.conf import settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from django.core.exceptions import ImproperlyConfigured


class RAG_defaults:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "BAAI/bge-small-en", 
                table_name: str = 'llama2_paper_v3', 
                embed_dim: int = 384):
        
        with cls._lock:  # Ensures thread safety
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._embedding_model = None
                cls._instance._vector_store = None
                cls._instance.model_name = model_name
                cls._instance.table_name = table_name
                cls._instance.embed_dim = embed_dim

            return cls._instance
        
    @staticmethod
    def _dev_load_env():
        """Loads the Django environment in development mode."""
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "personal_RAG.settings")
        with open(".secrets.json", "r") as f:
            secrets = json.load(f)
            for key, value in secrets.items():
                os.environ[key] = value

        import django
        django.setup()
    

    @property
    def embedding_model(self):
        """Lazy load the embedding model only when accessed."""
        if self._embedding_model is None:
            with self._lock:  # Thread-safe initialization
                if self._embedding_model is None:  # Double-check locking
                    self._embedding_model = HuggingFaceEmbedding(model_name=self.model_name)
        return self._embedding_model

    @property
    def vector_store(self) -> PGVectorStore:
        """Lazy load the vector store only when accessed."""
        if self._vector_store is None:
            with self._lock:  # Thread-safe initialization
                if self._vector_store is None:  # Double-check locking
                    try:
                        db_config = settings.DATABASES['default']
                    except ImproperlyConfigured:
                        self._dev_load_env()
                        db_config = settings.DATABASES['default']

                    self._vector_store = PGVectorStore.from_params(
                        database=db_config['NAME'],
                        host=db_config['HOST'],
                        password=db_config['PASSWORD'],
                        port=db_config['PORT'],
                        user=db_config['USER'],
                        table_name=self.table_name,
                        embed_dim=self.embed_dim,
                    )
        return self._vector_store


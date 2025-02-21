import threading
import os, json
from django.conf import settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from django.core.exceptions import ImproperlyConfigured
import re



class RAG_defaults:
    """Singleton class for default RAG settings."""
    _instance = None
    _lock = threading.Lock()
    required_metadata = ['user', 'project', 'filename', 'chapter']
    BASE_TABLE_NAME = "RAG_nodes_user_"

    def __new__(cls, model_name: str = "BAAI/bge-small-en", 
                table_name: str = 'llama2_paper_v3', 
                embed_dim: int = 384):
        
        with cls._lock:  # Ensures thread safety
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._embedding_model = None
                cls._instance._vector_stores = {}
                cls._instance.model_name = model_name
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

    def get_vector_store(self, user_id: int) -> PGVectorStore:
            """
            Returns a PGVectorStore instance specific to a user.
            The instance is cached in the singleton so that the DB session/table
            initialization occurs only once per user.
            """
            with self._lock:
                if user_id not in self._vector_stores:
                    try:
                        db_config = settings.DATABASES['default']
                    except ImproperlyConfigured:
                        self._dev_load_env()
                        db_config = settings.DATABASES['default']

                    # Sanitize the user_id to ensure it's safe for use in a table name
                    safe_user_id = re.sub(r'\W+', '_', str(user_id).lower())
                    table_name = self.BASE_TABLE_NAME + safe_user_id

                    self._vector_stores[user_id] = PGVectorStore.from_params(
                        database=db_config['NAME'],
                        host=db_config['HOST'],
                        password=db_config['PASSWORD'],
                        port=db_config['PORT'],
                        user=db_config['USER'],
                        table_name=table_name,
                        embed_dim=self.embed_dim,
                    )
                return self._vector_stores[user_id]

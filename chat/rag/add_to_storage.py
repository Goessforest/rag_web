from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from ._rag_defaults import RAG_defaults
from .file_to_markdown import FileToMarkdown

import datetime

RAG_defaults._dev_load_env()

CHUNK_SIZE = 512*2

class Parse_and_Store_Vector:
    def __init__(self, path:str):
        self.path = path
        # parse pdf to markdown
        self.text, self.name = FileToMarkdown().parseFile(path)

        # Setup defaults
        self.rag_defaults = RAG_defaults()
        self.nodes = []

        self.chunk_generator = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            # separator=" ",
        )
        # generate vectors and add to db
        self.generate_vectors()

    def generate_vectors(self):
        text_chunks = []
        # maintain relationship with source doc index, to help inject doc metadata in (3)

        cur_text_chunks = self.chunk_generator.split_text(self.text)
        text_chunks.extend(cur_text_chunks)


        # construct Nodes
        for idx, text_chunk in enumerate(text_chunks):
            node = TextNode(
                text=text_chunk,
            )
            node.metadata = self.metadata
            self.nodes.append(node)

        # create embaddings
        for node in self.nodes:
            node_embedding = self.rag_defaults.embedding_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        # add the nodes to the vectorstore
        self.rag_defaults.vector_store.add(self.nodes)

    @property
    def metadata(self) -> dict:
        """Generates the metadata dictionary for a file"""

        # Retrieve creation time (or fallback to modification time if not available)
        created_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Return metadata as a dictionary
        return {
            "filename": self.name,
            "format": "md",
            "created_time": created_time,
        }       
        

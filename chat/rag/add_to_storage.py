from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

from ._rag_defaults import RAG_defaults
from .file_to_markdown import FileToMarkdown
from llama_index.core.vector_stores import  MetadataFilters, MetadataFilter, FilterOperator

import datetime
import re, os
import hashlib
import logging


RAG_defaults._dev_load_env()

CHUNK_SIZE = 512
DEFAULT_CHAPTER_KEY = "Undefined"

class Parse_and_Store_Vector:
    def __init__(self, path:str, user_id:int, project_id:str=None):
        self.path = path
        self.user_id = user_id
        self.project_id = project_id

        # Setup defaults
        self.rag_defaults = RAG_defaults()
        self.nodes = []

        self.chunk_generator = SentenceSplitter(
            chunk_size=CHUNK_SIZE,
            # separator=" ",
        )
        # parse the file
        self.text, self.name = FileToMarkdown().parseFile(path)
        # generate vectors and add to db
        self.generate_vectors()

    @property
    def file_baseName(self):
        """returns the baseName of the path"""
        return os.path.basename(self.path)

    @property
    def file_id(self) -> str:
        """
        enerate a unique identifier for the file using a SHA-256 hash of the PDF file's binary content.

        Returns:
        - str: A hexadecimal string representing the hash.
        """
        if not hasattr(self, "_file_id"):
            sha256_hash = hashlib.sha256()
            if self.path.endswith(".pdf"):
                # read the file Content
                with open(self.path, "rb") as f:
                    # Read and update hash in chunks to handle large files
                    for byte_block in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(byte_block)
            else:
                sha256_hash.update(self.file_baseName.encode('utf-8'))
            self._file_id = sha256_hash.hexdigest()
        return self._file_id
    

    def get_file_type(self) -> str:
        """
        Estimate the type of literature.
        
        Returns:
        - "Scientific" if the text appears to be a scientific paper (e.g., includes sections like "Abstract", "Introduction", etc.).
        - "WebPage" if the text appears to be HTML content.
        - "Else" for other types of literature.
        """
        # Assume that the content is stored in self.text.
        if not hasattr(self, "text"):
            raise AttributeError("No text content available to determine file type.")

        content = self.text.lower()

        # Check for HTML markers
        if "<html" in content or "<!doctype html>" in content:
            return "WebPage"

        # Check for common scientific paper section headers.
        scientific_markers = [
            "abstract",
            "introduction",
            "methods",
            "materials and methods",
            "results",
            "discussion",
            "conclusion",
            "references",
        ]
        found_markers = sum(1 for marker in scientific_markers if marker in content)
        # Use a threshold: if at least two markers are found, we consider it a scientific paper.
        if found_markers >= 3:
            return "Scientific"

        # Fallback
        return "Else"



    def get_chapter(self, text_chunk: str) -> str:
        """
        Estimate the chapter/section of a scientific paper that contains the given text chunk.
        
        The function works by:
        1. Extracting candidate section headings (with multiple alias names) and their positions 
        from the full text. The recognized canonical groups are:
            - Abstract
            - Introduction (Thesis)
            - Literature review
            - Material and Methods
            - Results
            - Discussion
            - Conclusion
            - Bibliography
        2. Locating the position of the text_chunk within the full text.
        3. Determining the closest preceding section heading. If no recognized section is found
        before the text_chunk, "None Scientific" is returned.
        
        Parameters:
        - text_chunk (str): A snippet from the paper.
        
        Returns:
        - str: The estimated chapter/section name, or an appropriate message if not found.
        """
        
        # Define canonical sections and their alias regex patterns.
        alias_map = {
            "Abstract": [r'\bAbstract\b', r'\bSammary\b', r'\bZusammenfassung\b'],
            "Introduction (Thesis)": [r'\bIntroduction\b', r'\bPreface\b'],
            "Literature review": [r'\bLiterature review\b', r'\bRelated Work\b', r'\bReview of Literature\b', r'\bState of the art\b'],
            "Material and Methods": [r'\bMaterials? and Methods\b', r'\bMethodology\b', r'\bMethods\b'],
            "Results": [r'\bResults\b'],
            "Discussion": [r'\bDiscussion\b'],
            "Conclusion": [r'\bConclusion\b'],
            "Bibliography": [r'\bBibliography\b', r'\bReferences\b', r'\bWorks Cited\b', r'\bCited Works\b']
        }
        
        # Compile all alias patterns into a list of tuples (compiled_regex, canonical_name)
        compiled_patterns = []
        for canonical, patterns in alias_map.items():
            for pat in patterns:
                compiled_patterns.append((re.compile(pat, re.IGNORECASE), canonical))
        
        # Find all matches in the full text stored in self.text.
        sections = []
        for regex, canonical_name in compiled_patterns:
            for match in regex.finditer(self.text):
                # Append tuple (position, canonical name)
                sections.append((match.start(), canonical_name))
        
        # If no section headings found, return an informative message.
        if not sections:
            return DEFAULT_CHAPTER_KEY# "No section headings found in full text."
        
        # Sort the found sections by their position in the text.
        sections.sort(key=lambda x: x[0])
        
        # Locate the position of the text chunk in the full text.
        chunk_index = self.text.find(text_chunk)
        if chunk_index == -1:
            return DEFAULT_CHAPTER_KEY# "Text chunk not found in full text."
        
        # Determine the closest preceding section heading.
        estimated_section = None
        for pos, canonical_name in sections:
            if pos <= chunk_index:
                estimated_section = canonical_name
            else:
                break
        
        # If no section was found before the text chunk, label it as "None Scientific".
        return estimated_section if estimated_section else DEFAULT_CHAPTER_KEY



    def delete_dublicates(self): 
        """Delete dublicate nodes"""
        fileid_filter = MetadataFilter(
            key="file_id",
            value=self.file_id,
            operator=FilterOperator.EQ
        )        
        filterBundle = MetadataFilters(filters=[fileid_filter])
        self.rag_defaults.get_vector_store(self.user_id).delete_nodes(filters=filterBundle)
        logging.info(f"Deleted dublicates for file_id: {self.file_id}")

    

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
            node.metadata = self.metadata(text_chunk)
            self.nodes.append(node)

        # create embaddings
        for node in self.nodes:
            node_embedding = self.rag_defaults.embedding_model.get_text_embedding(
                node.get_content(metadata_mode="all")
            )
            node.embedding = node_embedding

        # add the nodes to the vectorstore
        self.delete_dublicates()
        self.rag_defaults.get_vector_store(self.user_id).add(self.nodes)


    def metadata(self, text_chunk:str) -> dict:
        """Generates the metadata dictionary for a file"""

        # Retrieve creation time (or fallback to modification time if not available)
        created_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Return metadata as a dictionary
        return {
            "project_id": str(self.project_id), 
            "filename": self.name,
            "file_id": self.file_id,
            "file_type": self.get_file_type(), # Estimate the type of literature
            "chapter": self.get_chapter(text_chunk), # Estimate the chapter/section
            "format": "md",
            "created_time": created_time
        }       
        

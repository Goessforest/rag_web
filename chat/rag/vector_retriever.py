from llama_index.core import QueryBundle
# from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from typing import Any, List, Optional
# import textwrap

import os
from ._rag_defaults import RAG_defaults
from ..open_ai.openai_query import OpenAIQuery, AI_Message
from typing import *


BASE_MESSAGE = """You are a scientific RAG (Retrieval-Augmented Generation) assistant. Your task is to analyze input information and provide answers strictly based on the provided context and reference sources. Follow these rules:
                1. Use only the information explicitly or implicitly present in the input and references. Precisly highlight implicit inference. Do not infer or add external knowledge or assumptions.
                2. Maintain the scientific integrity and meaning of the input. Avoid altering the context or intent of the references.
                3. If a question cannot be answered based on the provided input, explicitly state: "The provided sources do not contain information to answer this question. Before stating what related information the research provides."
                4. Cite the relevant parts of the input or reference sources that support your answer.
                5. Use concise, accurate, and formal scientific language.

                Ensure that all responses are consistent with the provided data and adhere to scientific rigor.
            """



class VectorDBRetriever():
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        query_mode: str = "default",
    ) -> None:
        """Init params."""
        self.rag_defaults = RAG_defaults()
        self._query_mode = query_mode
        self._openai_query = OpenAIQuery()
        super().__init__()


    def query(self, question: str, similarity_top_k:int=5, max_tokens:int=100) -> Union[NodeWithScore, AI_Message]:
        """Ask a question and get a response from Openai based on the query"""
        queryObject = QueryBundle(query_str=question)
        nodes_with_scores = self._retrieve(queryObject, similarity_top_k=similarity_top_k)

        response = self._openai_query.query(messages=self._create_messages(nodes_with_scores, queryObject), prompt=None, max_tokens=max_tokens)
        return nodes_with_scores, response


    def follow_up_query(self, question: str, similarity_top_k:int=5, max_tokens:int=10) -> str:
        """Ask a question followup and get a response from Openai based on the query"""
        queryObject = QueryBundle(query_str=question)
        nodes_with_scores = self._retrieve(queryObject, similarity_top_k=similarity_top_k)

        response = self._openai_query.follow_up_query(messages=self._create_messages(nodes_with_scores, queryObject), prompt=None, max_tokens=max_tokens)
        return nodes_with_scores, response



    def _create_messages(self, nodes_with_scores: str, queryObject: str):
        """Create the messages for the openai chat"""
        messages = [AI_Message(role="system", content=BASE_MESSAGE)]
        # context
        context = "Context: "
        for index, node in enumerate(nodes_with_scores):
            content = node.get_content().replace('\n', ' ')
            fileName = node.metadata.get('filename', 'Not Found')
            context += f"Source {index+1}: {fileName}, Score {node.score:.3f}: {content}\n ----- \n"
        messages.append(AI_Message(role= "user", content= context))
        # question
        messages.append(AI_Message(role= "user", content= f"{queryObject.query_str}"))
        return messages



    def _retrieve(self, query_bundle: QueryBundle, similarity_top_k:int=5) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self.rag_defaults.embedding_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self.rag_defaults.vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores


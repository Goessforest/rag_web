from llama_index.core import QueryBundle
# from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import VectorStoreQuery
from typing import Any, List, Optional
# import textwrap
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from llama_index.vector_stores.postgres import PGVectorStore

import os


class VectorDBRetriever():
    """Retriever over a postgres vector store."""

    def __init__(
        self,
        vector_store: PGVectorStore,
        embed_model: Any,
        query_mode: str = "default",
        similarity_top_k: int = 5,
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        self.openai_clinet = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_messages = [{
            "role": "system", 
            "content": """You are a scientific RAG (Retrieval-Augmented Generation) assistant. Your task is to analyze input information and provide answers strictly based on the provided context and reference sources. Follow these rules:
                1. Use only the information explicitly or implicitly present in the input and references. Precisly highlight implicit inference. Do not infer or add external knowledge or assumptions.
                2. Maintain the scientific integrity and meaning of the input. Avoid altering the context or intent of the references.
                3. If a question cannot be answered based on the provided input, explicitly state: "The provided sources do not contain information to answer this question. Before stating what related information the research provides."
                4. Cite the relevant parts of the input or reference sources that support your answer.
                5. Use concise, accurate, and formal scientific language.

                Ensure that all responses are consistent with the provided data and adhere to scientific rigor.
"""}
            # {"role": "user", "content": f"Context: {context}"},
            # {"role": "user", "content": f"Question: {question}"}
        ]
        super().__init__()

    def query(self, question: str) -> str:
        """Ask a question and get a response from Openai based on the query"""
        queryObject = QueryBundle(query_str=question)
        nodes_with_scores = self._retrieve(queryObject)

        response = self._ask_openai(messages=self._create_messages(nodes_with_scores, queryObject))
        return nodes_with_scores, response



    def _create_messages(self, nodes_with_scores: str, queryObject: str):
        """Create the messages for the openai chat"""
        messages = self.base_messages.copy()
        # context
        context = "Context: "
        for index, node in enumerate(nodes_with_scores):
            content = node.get_content().replace('\n', ' ')
            fileName = node.metadata.get('filename', 'Not Found')
            context += f"Source {index+1}: {fileName}, Score {node.score:.3f}: {content}\n ----- \n"
        messages.append({"role": "user", "content": context})
        # question
        messages.append({"role": "user", "content": f"{queryObject.query_str}"})
        return messages


    def _ask_openai(self, messages:list) -> ChatCompletion:
        """Sends the data to openai and returns the response"""
        response = self.openai_clinet.chat.completions.create(
                messages=messages,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

        return response # response["choices"][0]["message"]["content"].strip()


    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve."""
        query_embedding = self._embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = self._vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score: Optional[float] = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))
        return nodes_with_scores


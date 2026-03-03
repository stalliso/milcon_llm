import json
import os
import requests
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAI
from typing import Optional, Tuple
from langchain_chroma import Chroma

try:
    from .load_dataset import load_vectorstore
except ImportError:
    from load_dataset import load_vectorstore


@dataclass
class TraceEvent:
    """Single event in the RAG application trace"""

    event_type: str
    component: str
    data: Dict[str, Any]


class BaseRetriever:
    """
    Base class for retrievers.
    Subclasses should implement the fit and get_top_k methods.
    """

    def __init__(self):
        self.documents = []

    def fit(self, documents: List[str]):
        """Store the documents"""
        self.documents = documents

    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Retrieve top-k most relevant documents for the query."""
        raise NotImplementedError("Subclasses should implement this method.")


class SemanticRetriever(BaseRetriever):
    """Semantic retriever"""

    def __init__(self, vectorstore: Chroma, k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        self.vectorstore = vectorstore
        self.k = k


    def get_top_k(self, query: str, k: int = 3) -> List[tuple]:
        """Get top k documents by semantic similarity"""
        assert isinstance(query, str), "Your search query must be a string"

        # Use similarity_search_with_score to get documents with scores
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=k)

        # Return list of tuples: (document_content, similarity_score)
        return [(doc.page_content, score) for doc, score in docs_with_scores]


class ExampleRAG:
    """
    Simple RAG system that:
    1. accepts a llm client
    2. Uses semantic search to retrieve chunks
    3. uses the llm client to generate a response based on the retrieved documents when a query is made
    """

    def __init__(
        self,
        llm_client,
        model_id,
        retriever: Optional[BaseRetriever] = None,
        system_prompt: Optional[str] = None,
        logdir: str = "logs",
    ):
        """
        Initialize RAG system

        Args:
            llm_client: LLM client with a generate() method
            retriever: Document retriever (defaults to SimpleKeywordRetriever)
            system_prompt: System prompt template for generation
            logdir: Directory for trace log files
        """
        self.llm_client = llm_client
        self.model_id = model_id
        self.retriever = retriever
        self.system_prompt = (
            system_prompt
            or """Answer the following question based on the provided documents:
                                Question: {query}
                                Documents:
                                {context}
                                Answer:
                            """
        )
        self.documents = []
        self.is_fitted = False
        self.traces = []
        self.logdir = logdir

        # Create log directory if it doesn't exist
        os.makedirs(self.logdir, exist_ok=True)

        # Initialize tracing
        self.traces.append(
            TraceEvent(
                event_type="init",
                component="rag_system",
                data={
                    "retriever_type": type(self.retriever).__name__,
                    "system_prompt_length": len(self.system_prompt),
                    "logdir": self.logdir,
                },
            )
        )

   
    def retrieve_documents(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve top-k most relevant documents for the query

        Args:
            query: Search query
            top_k: Number of documents to retrieve

        Returns:
            List of dictionaries containing document info
        """
        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_start",
                    "query": query,
                    "query_length": len(query),
                    "top_k": top_k,
                },
            )
        )

        top_docs = self.retriever.get_top_k(query, k=top_k)

        retrieved_docs = []
        for idx, (content, score) in enumerate(top_docs):
            retrieved_docs.append(
                {
                    "content": content,
                    "similarity_score": score,
                    "document_id": idx,
                }
            )

        self.traces.append(
            TraceEvent(
                event_type="retrieval",
                component="retriever",
                data={
                    "operation": "retrieve_complete",
                    "num_retrieved": len(retrieved_docs),
                    "scores": [doc["similarity_score"] for doc in retrieved_docs],
                    "document_ids": [doc["document_id"] for doc in retrieved_docs],
                },
            )
        )

        return retrieved_docs

    def generate_response(self, query: str, top_k: int = 3) -> str:
        """
        Generate response to query using retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            Generated response
        """
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_documents(query, top_k)

        if not retrieved_docs:
            return "I couldn't find any relevant documents to answer your question."

        # Build context from retrieved documents
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}")

        context = "\n\n".join(context_parts)

        # Generate response using LLM client
        prompt = self.system_prompt.format(query=query, context=context)

        self.traces.append(
            TraceEvent(
                event_type="llm_call",
                component="openai_api",
                data={
                    "operation": "generate_response",
                    "model": self.model_id,
                    "query": query,
                    "prompt_length": len(prompt),
                    "context_length": len(context),
                    "num_context_docs": len(retrieved_docs),
                },
            )
        )

        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
            )

            response_text = response.choices[0].message.content.strip()

            self.traces.append(
                TraceEvent(
                    event_type="llm_response",
                    component="openai_api",
                    data={
                        "operation": "generate_response",
                        "response_length": len(response_text),
                        "usage": (
                            response.usage.model_dump() if response.usage else None
                        ),
                        "model": self.model_id,
                    },
                )
            )

            return response_text

        except Exception as e:
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="openai_api",
                    data={"operation": "generate_response", "error": str(e)},
                )
            )
            return f"Error generating response: {str(e)}"

    def query(
        self, question: str, top_k: int = 3, run_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Complete RAG pipeline: retrieve documents and generate response

        Args:
            question: User question
            top_k: Number of documents to retrieve
            run_id: Optional run ID for tracing (auto-generated if not provided)

        Returns:
            Dictionary containing response and retrieved documents
        """
        # Generate run_id if not provided
        if run_id is None:
            run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(question) % 10000:04d}"

        # Reset traces for this query
        self.traces = []

        self.traces.append(
            TraceEvent(
                event_type="query_start",
                component="rag_system",
                data={
                    "run_id": run_id,
                    "question": question,
                    "question_length": len(question),
                    "top_k": top_k,
                    "total_documents": len(self.documents),
                },
            )
        )

        try:
            retrieved_docs = self.retrieve_documents(question, top_k)
            response = self.generate_response(question, top_k)

            result = {"answer": response, "run_id": run_id}

            self.traces.append(
                TraceEvent(
                    event_type="query_complete",
                    component="rag_system",
                    data={
                        "run_id": run_id,
                        "success": True,
                        "response_length": len(response),
                        "num_retrieved": len(retrieved_docs),
                    },
                )
            )

            logs_path = self.export_traces_to_log(run_id, question, result)
            return {"answer": response, "run_id": run_id, "logs": logs_path}

        except Exception as e:
            self.traces.append(
                TraceEvent(
                    event_type="error",
                    component="rag_system",
                    data={"run_id": run_id, "operation": "query", "error": str(e)},
                )
            )

            # Return error result
            logs_path = self.export_traces_to_log(run_id, question, None)
            return {
                "answer": f"Error processing query: {str(e)}",
                "run_id": run_id,
                "logs": logs_path,
            }

    def export_traces_to_log(
        self,
        run_id: str,
        query: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None,
    ):
        """Export traces to a log file with run_id"""
        timestamp = datetime.now().isoformat()
        log_filename = (
            f"rag_run_{run_id}_{timestamp.replace(':', '-').replace('.', '-')}.json"
        )
        log_filepath = os.path.join(self.logdir, log_filename)

        log_data = {
            "run_id": run_id,
            "timestamp": timestamp,
            "query": query,
            "result": result,
            "num_documents": len(self.documents),
            "traces": [asdict(trace) for trace in self.traces],
        }

        with open(log_filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"RAG traces exported to: {log_filepath}")
        return log_filepath


def default_rag_client(llm_client, model_id, vectorstore_path,  logdir: str = "logs") -> ExampleRAG:
    """
    Create a default RAG client with OpenAI LLM and optional retriever.

    Args:
        retriever: Optional retriever instance (defaults to SimpleKeywordRetriever)
        logdir: Directory for trace logs
    Returns:
        ExampleRAG instance
    """
    vectorstore = load_vectorstore(vectorstore_path)
    retriever = SemanticRetriever(vectorstore)
    client = ExampleRAG(llm_client=llm_client, model_id=model_id, retriever=retriever, logdir=logdir)
    return client

if __name__ == "__main__":
    ENDPOINT_URL = "http://trac-malenia.ern.nps.edu:8080/gpu1/v1"
    VECTOR_STORE_PATH = "../data/databases/L5_chroma_db"
    LOG_DIR = "./logs/L5"
    
    try:
        response = requests.get(f"{ENDPOINT_URL}/models")
        response.raise_for_status()  # Raise an error for bad status codes
        info = response.json()
        repo_id = info['data'][0]['id']

        # verify
        print(f"Starting with locally hosted model: {repo_id}\n")
    except Exception as e:
        print(f"Error: {e}")

    # Initialize RAG system with tracing enabled
    llm = OpenAI(
        base_url= ENDPOINT_URL, # defaults to /v1/chat/completions endpoint
        api_key = "dummy",  # vllm doesn't require a real key
    )

    vectorstore = load_vectorstore(VECTOR_STORE_PATH)
    r = SemanticRetriever(vectorstore)
    rag_client = ExampleRAG(llm_client=llm, model_id=repo_id, retriever=r, logdir=LOG_DIR)


    # Run query with tracing
    query = "Explain how a fire mission works."
    print(f"Query: {query}")
    response = rag_client.query(query, top_k=3)

    print("Response:", response["answer"])
    print(f"Run ID: {response['logs']}")

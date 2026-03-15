# ____________________________________________________________________________________________________________________________________________
#
#                                               MILCON LLM WORKFLOW SCRIPT
#                                            (Single-Router/Retriever Version)
#
#____________________________________________________________________________________________________________________________________________________

# IMPORTS:
#____________________________________________________________________________________________________________________________________________________
# Standard library
import ast
import asyncio
import json
import logging
import operator
import os
import re
import sys
import textwrap
import time
import warnings
from collections import defaultdict
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from typing import Annotated, List, Literal
import csv, datetime

# Third-party
import numpy as np
import requests
from IPython.display import Image, display
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
from tqdm import tqdm
from typing_extensions import NotRequired, TypedDict

# OpenAI
from openai import APIConnectionError, AsyncOpenAI, OpenAI

# LangChain
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer

# LangGraph
from langgraph.graph import END, StateGraph

# Project - eval
from eval.eval_multi_part import eval_multi_part_routing
from eval.eval_task1_quant import (
    baseline_hfe,
    baseline_rag_chain,
    baseline_retriever,
    baseline_vectorstore,
    eval_rag_chain_proj_query,
    llm,
)
from concurrent.futures import ThreadPoolExecutor, as_completed 

from eval.eval_multi_part import EXPECTED_ROUTES, MULTI_PART_QUESTIONS

# Project - helper code
sys.path.append(str(Path.cwd().parent))
from helper_code.build_vectorstores import build_vectorstores
from helper_code.parser import *
from helper_code.rag.load_dataset import load_db_from_dir, load_vectorstore, setup_embedding_function
from helper_code.export_eval import export_eval_summary_png                   #
from eval.eval_multi_part import MULTI_PART_QUESTIONS, EXPECTED_ROUTES

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

baseline_hfe = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# Or just check the directory itself is non-empty
def db_exists(path):
    p = Path(path)
    return p.exists() and any(p.iterdir())

if not all(db_exists(p) for p in ["./databases/proj", "./databases/strat26", "./databases/strat28"]):

    print("One or more vectorstores are missing. Please run milcon_doc_vectorstore.py to create them.")
    build_vectorstores()

# Load vectorstores:
proj_vectorstore_path: str = "databases/proj"
proj_vectorstore = load_vectorstore(proj_vectorstore_path, embedding_function=baseline_hfe)

strat26_vectorstore_path: str = "./databases/strat26"
strat26_vectorstore = load_vectorstore(strat26_vectorstore_path, embedding_function=baseline_hfe)

strat28_vectorstore_path: str = "./databases/strat28"
strat28_vectorstore = load_vectorstore(strat28_vectorstore_path, embedding_function=baseline_hfe)

# Cross-encoder for reranking (hybrid retrieval + rerank)
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
#____________________________________________________________________________________________________________________________________________________

# INITIALIZE LLMS:
#____________________________________________________________________________________________________________________________________________________

API_KEY = "sk-UtrV9i5fFenmG6hvMss71A"
BASE_URL = "http://trac-malenia.ern.nps.edu:8080/inference/v1"

# Fallback when endpoint is unreachable (e.g. offline, no VPN)
DEFAULT_MODEL_IDS = ["meta/llama-3.2-3b-instruct", "meta/llama-3.2-3b-instruct", "meta/llama-3.2-3b-instruct"]

# Check the models available
model_ids = []

try:
    response = requests.get(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=10,
    )
    response.raise_for_status()
    info = response.json()
    for model in info["data"]:
        model_ids.append(model["id"])
    print(f"Available Models: {model_ids}")
except Exception as e:
    print(f"Error accessing model endpoint: {e}")
    print("  (If models are on NPS campus, check VPN/network and that trac-malenia.ern.nps.edu is reachable.)")

if len(model_ids) < 3:
    model_ids = DEFAULT_MODEL_IDS.copy()
    print(f"Using fallback model IDs: {model_ids}")

model_id = model_ids[2]

########## ROUTER MODEL ##########
llm_md_tools = ChatOpenAI(
    base_url=BASE_URL,
    model=model_id,
    api_key=API_KEY,
    temperature=0,  
    name= "llama" # Name the LLM for langchain
)

########## GENERATION MODEL ##########
llm_gen_tools = ChatOpenAI(
    base_url=BASE_URL,
    model= model_ids[1],
    api_key=API_KEY,
    temperature=0.3,
    name="llama"
)

########## GRADER MODEL ##########
llm_grader = ChatOpenAI(
    base_url=BASE_URL,
    model=model_ids[1],  # google/gemma-3-27b-it
    api_key=API_KEY,
    temperature=0,
    name="grader"
)

grader_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a grader assessing whether an answer is grounded in the retrieved documents.
Respond with ONLY 'yes' or 'no'. No explanation.

- 'yes': The answer is supported by the retrieved context and directly addresses the question.
- 'no': The answer contains information not in the context, says 'I don't know', or fails to address the question."""),
    ("human", """Retrieved Context:
{context}

Question: {question}

Answer: {generation}

Is the answer grounded in the context?""")
])

grader_chain = grader_prompt | llm_grader | StrOutputParser()
#____________________________________________________________________________________________________________________________________________________

# LOGGER SETUP:
#____________________________________________________________________________________________________________________________________________________
# Setup logging for tool creation and testing
logger = logging.getLogger("agentic_workflow")
logger.setLevel(logging.DEBUG)

# Add console handler if not already present
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Logger configured for tool creation and testing")

# create logging runnable to track router chain execution
def logging_helper(state: dict) -> dict:
    """Log internal chain steps"""
    logger.debug(f"Intermediary State: '{state}'")
    return state

r_logger = RunnableLambda( logging_helper, name= "log chain state")
#____________________________________________________________________________________________________________________________________________________



# BUILD THE ROUTER:
#____________________________________________________________________________________________________________________________________________________
## YELLOW ##
# Provide instructions for the routing tool, and the available routes. See Lsn8 for example.
# Instructions should be something like use this vectorstore for this, use that one for that, otherwise use...
class RouteSelection(BaseModel):
    """Schema for selecting one or more relevant vectorstores.
    """
    routes: List[Literal["proj_vectorstore", "strat26_vectorstore", "strat28_vectorstore"]] = Field(
        description="Selected route based on query type."
    )

@tool(args_schema=RouteSelection)
def select_route(routes: list[str]) -> list[str]:
    """Return the list of vectorstores selected by the router.
    """
    return routes

llm_router = llm_md_tools.bind_tools([select_route], tool_choice='select_route')

# Router instructions
router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a routing assistant. Your task is to determine which document domains are relevant to the user’s query. 
Return ALL applicable vectorstores as a list. Queries may require multiple domains.

DOMAIN DEFINITIONS
- proj_vectorstore: Project datasheets and planning documents. Includes scope, cost, location, justification, 
  scoring (Mission Alignment, Readiness Support, Operational Cost, Severity/Urgency), facility details, and 
  proponent information.

- strat26_vectorstore: POM26-era strategic guidance. Includes NSS/NDS themes, CNIC scoring criteria, and 
  definitions for PDS scoring categories as they existed during POM26.

- strat28_vectorstore: Updated POM28 strategic guidance. Includes revised NSS/NDS themes, updated CNIC scoring 
  criteria, and new definitions for PDS scoring categories used for re-scoring or estimating projects under 
  updated guidance.

ROUTING RULES
- Include proj_vectorstore for any query involving a specific project ID (e.g., P738, RM16-0799), project 
  attributes, scoring, justification, or facility details.

- Include strat26_vectorstore for questions about POM26 strategy, POM26 scoring definitions, or “existing” or 
  “original” guidance.

- Include strat28_vectorstore for questions about POM28 strategy, updated scoring definitions, or requests to 
  estimate or reinterpret a project using “new,” “updated,” or “future” guidance.

- If a query asks how a project aligns with strategy or policy, include BOTH the project vectorstore and the 
  relevant strategy vectorstore(s).

OUTPUT FORMAT
Return a list of all relevant vectorstores, e.g.:
["proj_vectorstore", "strat28_vectorstore"]

"""),
    ("human", "{query}")
])

# Create the router chain
router_chain = router_prompt | r_logger | llm_router

# Test -- route should = proj_vectorstore
#result = router_chain.invoke("Are there any airfield projects in Italy?") # proj_vectorstore
#result = router_chain.invoke("How do players determine if a move is legal in Dungeons and Dragons?") # scen_vectorstore
#result = router_chain.invoke("Where can I buy the materials needed to play Dungeons and Dragons?") # websearch
#print(result)
#print(result.tool_calls[0]["args"])
#____________________________________________________________________________________________________________________________________________________

# LINK TOOLS TOGETHER WITH LANGGRAPH:
#____________________________________________________________________________________________________________________________________________________
class GraphState(TypedDict):
    """State for the agentic RAG workflow.

    Only 'question' is required. All other fields have defaults via init_graph_state().
    """
    question: str
    route: str | None
    routes: NotRequired[list[str]]
    generation: NotRequired[str]
    max_retries: NotRequired[int]
    gen_attempts: NotRequired[Annotated[int, operator.add]]
    documents: NotRequired[list[tuple[Document, int]]]
    k: NotRequired[int]
    route_after_check: NotRequired[str]
#____________________________________________________________________________________________________________________________________________________

# BUILD GRAPH NODES:
#____________________________________________________________________________________________________________________________________________________
# Build the initial graph state node
def init_graph_state(state: dict) -> GraphState:
    """Initialize GraphState with defaults from partial input.
    
    Takes a dict with at minimum 'question' key and fills in defaults for other fields.
    """
    logger.info("NODE: Initialize State")
    logger.debug(f"  Input state: {state}")
    
    # Fill in defaults for any missing keys
    initialized: GraphState = {
        "question": state["question"],
        "route": state.get("route", None),
        "routes": state.get("routes", []),
        "generation": state.get("generation", ""),
        "max_retries": state.get("max_retries", 3),
        "gen_attempts": state.get("gen_attempts", 0),
        "documents": state.get("documents", []),
        "k": state.get("k", 3),
    }
    
    logger.debug(f"  Initialized state keys: {list(initialized.keys())}")
    return initialized



# Build route question node
def route_question_node(state: dict) -> dict:
    """Determine routing decision and persist it in state"""
    logger.info("NODE: Route Question - Determining data source")

    question = state["question"]
    logger.debug(f"  Question to route: {question}")

    result = router_chain.invoke({"query": question})
    routes = result.tool_calls[0]["args"]["routes"]

    logger.info(f"  Routing decision: {routes}")
    return {"routes": routes}

# Build the retrieval node
def extract_project_id(question: str) -> str | None:
    match = re.search(
        r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4}|B\d{3,4})\b',
        question,
        re.IGNORECASE
    )
    return match.group(0).upper() if match else None


# Constants for is_aggregate_query (used to avoid resolving project from history on aggregate questions)
AGGREGATE_KEYWORDS = [
    "how many", "count", "total cost", "total amount", "total cwe", "combined cost",
    "sum of", "list all", "show all", "all projects", "which projects", "average cost",
    "avg cost", "projects in", "projects across",
]
SPECIFIC_PROJECT_PATTERNS = [
    r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4}|B\d{3,4})\b',
]
NARRATIVE_KEYWORDS = [
    "justification", "justify", "explain", "why", "how does", "align with", "strategy",
    "strategic", "guidance", "scope", "impact", "description", "what is the",
    "tell me about", "summarize",
]


def is_aggregate_query(question: str) -> bool:
    """Return True only for dataset-wide summary/count/filter queries."""
    if "Current question:" in question:
        q = question.split("Current question:")[-1].strip().lower()
    else:
        q = question.lower()
    for pattern in SPECIFIC_PROJECT_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return False
    if any(kw in q for kw in NARRATIVE_KEYWORDS):
        return False
    return any(kw in q for kw in AGGREGATE_KEYWORDS)


def extract_project_id_from_history(question: str) -> str | None:
        """
        Extract project ID from conversation history when current question
        uses a pronoun like 'it', 'this project', 'the project' without naming it.
        Scans the full enriched question string (which includes history) for the
        most recently mentioned project ID.
        """
        # Find ALL project IDs mentioned anywhere in the full question string
        # (which includes conversation history when enriched)
        all_matches = re.findall(
            r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4}|B\d{3,4})\b',
            question,
            re.IGNORECASE
        )
        if all_matches:
            # Return the LAST mentioned project ID — most recent in conversation
            return all_matches[-1].upper()
        return None
        
# Helper function for hybrid sparse and dense retrieval
# Sparse excels with more exact matches of tokens in user prompt, dense excels at finding matches based on semantic meaning
def hybrid_retrieve(store, query, k=6, filter_dict = None):

    # Dense retrieval
    dense_docs = store.similarity_search_with_relevance_scores(
        query, k=k, filter=filter_dict
    )

    # Sparse retrieval
    collection = store._collection.get(include=["documents", "metadatas"])
    bm25_docs = [
        Document(page_content = doc, metadata = meta)
        for doc, meta in zip(collection["documents"], collection["metadatas"])
    ]
    sparse_retriever = BM25Retriever.from_documents(documents=bm25_docs)
    sparse_docs = sparse_retriever.invoke(query)

    # Normalize scores
    def normalize(arr):
        arr = np.array(arr)
        return (arr - arr.min())/(arr.max() - arr.min() + 1e-6)
    
    sparse_scores = normalize([1.0] * len(sparse_docs))

    # Fuse scores
    fused = []
    for (doc, d_score), s_score in zip(dense_docs, sparse_scores):
        fused_score = 0.6 * d_score + 0.4 * s_score  # Weight the dense score slightly more than the sparse score
        fused.append((doc, float(fused_score)))
    
    for doc in sparse_docs[len(dense_docs):]:
        fused.append((doc, 0.4))
    
    # Sort and return
    fused_sorted = sorted(fused, key= lambda x: x[1], reverse = True)
    return fused_sorted[:k]

# Cross-Encoder Re-ranking helper function
def cross_encoder_rerank(query: str, docs_with_scores: list, top_k: int = 6):
    '''
    Input: List of (Doc, score) pairs from hybrid retrieval
    Returns: List of (Doc, new_score) pairs sorted by cross-encoder relevance
    '''
    # Set up query, doc_text pairs
    pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]

    # Get new scores predicted by the cross-encoder model
    scores = cross_encoder.predict(pairs)

    # Zip the docs with their new scores
    reranked = list(zip([doc for doc, _ in docs_with_scores], scores))

    # Sort by score descending
    reranked_sorted = sorted(reranked, key = lambda x: x[1], reverse = True)

    return [(doc, float(score)) for doc, score in reranked_sorted[:top_k]]


def semantic_retrieve_w_scores(state: dict) -> dict:
        logger.info("NODE: Semantic Retrieve - Starting retrieval")
        routes = state.get("routes", [])
        full_question = state["question"]
        k = state.get("k", 6)

        if "Current question:" in full_question:
            retrieval_query = full_question.split("Current question:")[-1].strip()
        else:
            retrieval_query = full_question

        # Resolve project ID BEFORE any query rewrite
        # so pronoun references ("it", "the project") are resolved from history
        project_id = extract_project_id(retrieval_query)

        PRONOUN_REFS = ["it ", "its ", "this project", "the project", "how does it", "how would it"]
        is_pronoun_ref = any(p in retrieval_query.lower() for p in PRONOUN_REFS)

        if not project_id and is_pronoun_ref and not is_aggregate_query(full_question):
            project_id = extract_project_id_from_history(full_question)
            if project_id:
                logger.info(f"  project_id resolved from history: {project_id}")

        # If a project was resolved (either directly or from history),
        # make sure proj_vectorstore is in routes
        if project_id and "proj_vectorstore" not in routes:
            routes = list(routes) + ["proj_vectorstore"]
            logger.info(f"  Safety net: added proj_vectorstore for {project_id}")

        # NOW apply query rewrite for NSS/NDS — after project ID is resolved
        # NOW apply query rewrite for NSS/NDS — after project ID is resolved
        q_lower = retrieval_query.lower()
        full_q_lower = full_question.lower()

        # Determine which POM era is being asked about
        is_pom28_context = any(kw in q_lower for kw in [
            "pom28", "pom 28", "updated", "new guidance", "updated guidance"
        ]) or any(kw in full_q_lower.split("current question:")[-1].lower() for kw in [
            "pom28", "pom 28"
        ])

        if any(kw in q_lower for kw in ["nss", "national security strategy"]):
            if is_pom28_context:
                retrieval_query = "2025 national security strategy america first priorities strength deterrence"
                logger.info("  NSS query rewrite applied (POM28 context)")
            else:
                retrieval_query = "national security strategy priorities competition democracy alliances"
                logger.info("  NSS query rewrite applied (POM26 context)")
        elif any(kw in q_lower for kw in ["nds", "national defense strategy"]):
            if is_pom28_context:
                retrieval_query = "2026 national defense strategy military strength deterrence homeland"
                logger.info("  NDS query rewrite applied (POM28 context)")
            else:
                retrieval_query = "national defense strategy military deterrence threats integrated"
                logger.info("  NDS query rewrite applied (POM26 context)")

        filter_dict = {"project_id": {"$eq": project_id}} if project_id else None
        logger.info(f"  project_id: {project_id}, filter: {filter_dict}")

        store_map = {
            "proj_vectorstore": proj_vectorstore,
            "strat26_vectorstore": strat26_vectorstore,
            "strat28_vectorstore": strat28_vectorstore
        }

        stores = [store_map[r] for r in routes if r in store_map]
        logger.info(f"  Stores to query: {[r for r in routes if r in store_map]}")

        if not stores:
            return {"documents": []}

        all_docs = []
        for store in stores:
            filt = filter_dict if store is proj_vectorstore else None
            logger.info(f"  Querying store with filter: {filt}")
            docs = hybrid_retrieve(
                store = store,
                query = retrieval_query,
                k=k*3,                         # Get a larger pool of candidate docs to be reranked
                filter_dict = filt
            )
            logger.info(f"  Hybrid retrieved {len(docs)} docs from store")

            reranked_docs = cross_encoder_rerank(
                query = retrieval_query,
                docs_with_scores=docs,
                top_k=k                       # Rerank and keep the top k
            )
            all_docs.extend(reranked_docs)

        return {"documents": all_docs}

semantic_retriever = RunnableLambda(
    semantic_retrieve_w_scores,
    name="semantic_retriever"
)

rag_template = (
    ChatPromptTemplate.from_messages([
        ("system", """You answer questions using only the retrieved documents. 
Your role is to retrieve information from one or multiple document domains:

- Project documents: scope, cost, justification, scoring, facility details.
- POM26 strategy/policy: NSS/NDS themes and original CNIC scoring criteria.
- POM28 strategy/policy: updated NSS/NDS themes and revised CNIC scoring criteria.
- Facility criteria documents: definitions and scoring rules for PDS categories.

GROUNDING RULES
- Treat the retrieved documents as the authoritative source for all project, facility, and strategy content.
- Do not invent or infer information that is not supported by the documents or tool outputs.
- If the documents do not contain the information needed to answer the question, respond with “I don’t know.”

MISSING‑EVIDENCE RULES
- If no documents are retrieved and the question concerns Navy facilities projects, respond with “I don’t know.”
- If no documents are retrieved and the question is general (not about Navy facilities), you may answer using general knowledge.

ANSWERING RULES
- Cite or reference document content in natural language.
- When multiple document domains are retrieved, integrate them into a single coherent answer.
- When POM26 and POM28 guidance conflict, clearly distinguish between them.
         """),
        ("human", """Retrieved Documents:
{context}

Question: {query}
         
         {answer_instructions}

Answer using only the retrived documents.""")
    ])
    .with_config(run_name="RAG_Prompt_Template")
)

# Build generation node
def generate_response(state: dict) -> dict:
    """Generate LLM response and add it to the running state."""
    logger.info("NODE: Generate Response")
    question = state['question']
    documents = state['documents']
    # dice_roll = state.get('dice_roll')
    
    logger.debug(f"  Question: {question}")
    logger.debug(f"  Using {len(documents)} documents for context")
    # logger.debug(f"  Dice roll: {dice_roll}")

    context = "\n\n---\n\n".join([doc.page_content for doc, _ in documents])

    has_choices = bool(re.search(r'\b[A-D]\)', question))
    
    if has_choices:
        answer_instructions = (
            "Instructions: You MUST respond with ONLY one of the provided answer choices "
            "exactly as written (e.g. 'D) 4'). Do NOT make up an answer that is not one "
            "of the options. Do not include any explanation or additional text."
        )
    else:
        answer_instructions = (
            "Instructions: Answer the question clearly and concisely using only "
            "the provided documents. If the answer is not in the documents, say \"I don't know\"."
        )

    logger.debug(f"  Context preview: {context[:100]}...")
    logger.debug(f"  Context length: {len(context)} chars")
    
    # # Create the prompt from state
    prompt = rag_template.invoke({"query": question, "context": context, "answer_instructions": answer_instructions})
    logger.info(f"  Prompt created with {len(prompt.messages)} messages")
    logger.debug(f"  Formatted prompt messages: {[m.type for m in prompt.messages]}")
    
    # Invoke LLM and generate
    logger.info("  Invoking LLM...")
    msg = llm_gen_tools.invoke(prompt.messages)

    # Parse to string
    generation_text = getattr(msg, "content", "") or ""
    logger.info(f"  Generated response ({len(generation_text)} chars)")
    logger.debug(f"  Response preview: {generation_text[:200]}...")
    
    return {"generation": generation_text, "gen_attempts": 1}

generate_response_runnable = RunnableLambda(generate_response, name="generate_response")



# Node to check if generation based on retrieval contains idk
idk_phrases = [
    "I don't know",
    "I do not know",
]

def check_answer_node(state: dict) -> dict:
    logger.info("NODE: Check Answer")

    question    = state.get("question", "")
    generation  = state.get("generation", "")
    documents   = state.get("documents", [])
    attempts    = state.get("gen_attempts", 0)
    max_retries = state.get("max_retries", 3)

    # ---- IDK check ----
    if any(phrase.lower() in generation.lower() for phrase in idk_phrases):
        logger.info("  IDK detected")
        if attempts < max_retries:
            return {"route_after_check": "retry"}
        return {"route_after_check": "final"}

    # ---- exact match enforcement for multiple choice ----
    if bool(re.search(r'\b[A-D]\)', question)):
        if not re.search(r'\b[A-D]\) ?\S+', generation):
            logger.info("  No valid answer choice found")
            if attempts < max_retries:
                return {"route_after_check": "retry"}
        logger.info("  Multiple choice answer found -> final")
        return {"route_after_check": "final"}

    # ---- LLM grounding check for open-ended ----
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in documents[:4]])
    grade = grader_chain.invoke({
        "context":    context,
        "question":   question,
        "generation": generation,
    }).strip().lower()

    logger.info(f"  Grader verdict: {grade}")

    if grade == "no" and attempts < max_retries:
        return {"route_after_check": "retry"}

    return {"route_after_check": "final"}

#____________________________________________________________________________________________________________________________________________________

# BUILD GRAPH EDGES:
#____________________________________________________________________________________________________________________________________________________
# Route Question Edge
def route_question_edge(state: dict) -> str:
    logger.info("EDGE: route_question_edge -> retrieve")
    return "retrieve"

# Decide after checking generation edge:
def decide_after_check(state: dict) -> str:
    return state.get("route_after_check", "final")
#____________________________________________________________________________________________________________________________________________________

# BUILD WORKFLOW:
#____________________________________________________________________________________________________________________________________________________
# Create the graph
workflow = StateGraph(GraphState)

# Add the appropriate nodes
workflow.add_node("init_state", init_graph_state)
workflow.add_node("route_question", route_question_node)
workflow.add_node("semantic_retriever_and_grader", semantic_retrieve_w_scores)
workflow.add_node("generate_response", generate_response)
workflow.add_node("check_answer", check_answer_node)

# Start -> init_state
workflow.set_entry_point("init_state")

# Add edges
# init_state -> route_question
workflow.add_edge("init_state", "route_question")

# route_question -> websearch or retrieve (conditional)
workflow.add_conditional_edges("route_question", 
                  route_question_edge,
                  {
                    "retrieve": "semantic_retriever_and_grader" 
                  }
                  )

# retrieve -> generate
workflow.add_edge("semantic_retriever_and_grader", "generate_response")


# generate -> check_answer
workflow.add_edge("generate_response", "check_answer")

# check_answer -> websearch or END (conditional)
workflow.add_conditional_edges("check_answer",
                               decide_after_check,
                               {
                                   "final": END,
                                   "retry": "semantic_retriever_and_grader"
                               }
                               )
# Compile the graph
app = workflow.compile()

#____________________________________________________________________________________________________________________________________________________
# UPFRONT HEALTH CHECK (NPS API reachable before running eval)
#____________________________________________________________________________________________________________________________________________________
def check_nps_api_reachable() -> tuple[bool, str]:
    """Returns (success, message). Call before eval to avoid failing on first question."""
    try:
        r = requests.get(
            f"{BASE_URL}/models",
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if data.get("data") and len(data["data"]) > 0:
            return True, "NPS API reachable (models endpoint OK)."
        return False, "NPS API returned no models."
    except requests.exceptions.Timeout:
        return False, "NPS API timeout (no response within 15s)."
    except requests.exceptions.ConnectionError as e:
        return False, f"NPS API unreachable: {e}"
    except Exception as e:
        return False, f"NPS API check failed: {e}"

api_ok, api_msg = check_nps_api_reachable()
print(api_msg)
if not api_ok:
    print("  Check VPN/network and that trac-malenia.ern.nps.edu is reachable.")
    sys.exit(1)

#____________________________________________________________________________________________________________________________________________________
# EVALUATE:
#____________________________________________________________________________________________________________________________________________________
# Retry on transient server disconnects (campus network hiccups, NPS server load)
MAX_LLM_RETRIES = 5
RETRY_BASE_DELAY_SEC = 10  # exponential backoff: 10s, 20s, 40s, 80s


def _invoke_app_with_retry(question: str) -> str:
    last_err = None
    for attempt in range(MAX_LLM_RETRIES):
        try:
            return app.invoke({"question": question, "k": 6})["generation"]
        except APIConnectionError as e:
            last_err = e
            logging.warning("APIConnectionError on question=%r (attempt %d/%d): %s",
            question[:80], attempt + 1, MAX_LLM_RETRIES, e)
            if attempt == MAX_LLM_RETRIES - 1:
                raise
            delay = RETRY_BASE_DELAY_SEC * (2**attempt)
            logging.warning(
                "API connection error (attempt %d/%d), retrying in %ds...",
                attempt + 1, MAX_LLM_RETRIES, delay
            )
            time.sleep(delay)
        except Exception as e:
            if "RemoteProtocolError" in type(e).__name__ or "disconnected" in str(e).lower():
                last_err = e
                if attempt == MAX_LLM_RETRIES - 1:
                    raise
                delay = RETRY_BASE_DELAY_SEC * (2**attempt)
                logging.warning(
                    "Server disconnected (attempt %d/%d), retrying in %ds...",
                    attempt + 1, MAX_LLM_RETRIES, delay
                )
                time.sleep(delay)
            else:
                raise
    raise last_err

# Wrap app.invoke to match the expected interface (takes string, returns string)
agent_as_chain = RunnableLambda(_invoke_app_with_retry)

# Suppress debug logging during eval
logging.getLogger("agentic_workflow").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

num_qa_runs    = 10
num_multi_runs = 10

def run_qa(_):
    # now returns (accuracy, missed_questions)
    return eval_rag_chain_proj_query(agent_as_chain, q_num=10, verbose=False)

def run_multi(_):
    return eval_multi_part_routing(app, k=6, verbose=False)

# Use 2 workers so QA and multi-part can run together without overloading NPS API
with ThreadPoolExecutor(max_workers=1) as executor:
    qa_futures    = {executor.submit(run_qa,    i): i for i in range(num_qa_runs)}
    multi_futures = {executor.submit(run_multi, i): i for i in range(num_multi_runs)}

    qa_results_raw = []
    for f in tqdm(as_completed(qa_futures), total=num_qa_runs, desc="QA runs", unit="run"):
        qa_results_raw.append(f.result())

    multi_part_list = []
    for f in tqdm(as_completed(multi_futures), total=num_multi_runs, desc="Routing runs", unit="run"):
        multi_part_list.append(f.result())

# ---- unpack QA results ----
qa_accuracies  = [r[0] for r in qa_results_raw]
qa_miss_counts = defaultdict(int)
for _, missed in qa_results_raw:
    for q in missed:
        qa_miss_counts[q] += 1

# ---- unpack routing results ----
multi_part_results = {}
run_accuracies     = []
q_pass_counts      = defaultdict(int)

for run, multi_result in enumerate(multi_part_list):
    multi_part_results[run] = multi_result
    run_accuracy = sum(r["passed"] for r in multi_result.values()) / len(multi_result)
    run_accuracies.append(run_accuracy)
    for q_key, res in multi_result.items():
        if res["passed"]:
            q_pass_counts[q_key] += 1

# ---- summary ----
print(f"\n{'='*70}")
print(f"  RESULTS — QA over {num_qa_runs} runs, Routing over {num_multi_runs} runs")
print(f"{'='*70}")

print(f"\nQA Accuracy")
print(f"  Mean  : {np.mean(qa_accuracies):.2%}")
print(f"  Stdev : {np.std(qa_accuracies):.2%}")
print(f"  Min   : {np.min(qa_accuracies):.2%}")
print(f"  Max   : {np.max(qa_accuracies):.2%}")

print(f"\nMost Missed QA Questions  (out of {num_qa_runs} runs)")
if qa_miss_counts:
    for q, count in sorted(qa_miss_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>3}x  {q}")
else:
    print("  None — all questions answered correctly across all runs")

print(f"\nRouting Accuracy")
print(f"  Mean  : {np.mean(run_accuracies):.2%}")
print(f"  Stdev : {np.std(run_accuracies):.2%}")
print(f"  Min   : {np.min(run_accuracies):.2%}")
print(f"  Max   : {np.max(run_accuracies):.2%}")

print(f"\nPer-Question Routing Pass Rate  (out of {num_multi_runs} runs)")
for q_key in sorted(MULTI_PART_QUESTIONS.keys()):
    count = q_pass_counts.get(q_key, 0)
    print(f"  {q_key:<45}  {count}/{num_multi_runs}  ({count/num_multi_runs:.0%})")

missed_routing = [q for q in sorted(MULTI_PART_QUESTIONS.keys()) if q_pass_counts.get(q, 0) == 0]
if missed_routing:
    print(f"\nNever passed routing:")
    for q in missed_routing:
        print(f"  {q}  (expected: {sorted(EXPECTED_ROUTES[q])})")
#____________________________________________________________________________________________________________________________________________________

# BUILD WORKFLOW:
#____________________________________________________________________________________________________________________________________________________

# # Display Graph
# display(Image(app.get_graph().draw_mermaid_png()))

with open("outputs/graph.png", "wb") as f:
    f.write(app.get_graph().draw_mermaid_png())

export_eval_summary_png(                                               
    qa_accuracies=qa_accuracies,                                       
    run_accuracies=run_accuracies,                                     
    q_pass_counts=q_pass_counts,                                       
    qa_miss_counts=qa_miss_counts,                                     
    multi_part_questions=MULTI_PART_QUESTIONS,                         
    expected_routes=EXPECTED_ROUTES,                                   
    num_multi_runs=num_multi_runs,                                     
    num_qa_questions=10,                                               
    output_path="outputs/eval_summary.png",                                    
)   

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# QA per-run
with open(f"outputs/qa_results_{timestamp}.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["run", "accuracy"])
    writer.writeheader()
    writer.writerows([{"run": i, "accuracy": acc} for i, acc in enumerate(qa_accuracies)])

# Missed QA questions
with open(f"outputs/qa_missed_{timestamp}.csv", "w", newline="") as f:
    rows = [{"question": q, "miss_count": count, "runs": num_qa_runs,
             "miss_rate": count / num_qa_runs}
            for q, count in sorted(qa_miss_counts.items(), key=lambda x: -x[1])]
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

# Routing per-question
with open(f"outputs/routing_results_{timestamp}.csv", "w", newline="") as f:
    rows = [{"question": q, "passes": q_pass_counts.get(q, 0), "runs": num_multi_runs,
             "pass_rate": q_pass_counts.get(q, 0) / num_multi_runs,
             "expected_routes": "|".join(sorted(EXPECTED_ROUTES[q]))}
            for q in sorted(MULTI_PART_QUESTIONS.keys())]
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


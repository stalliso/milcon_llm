#____________________________________________________________________________________________________________________________________________________
#
#                                               MILCON LLM WORKFLOW SCRIPT
#                                            (Single-Router/Retriever Version)
#
#____________________________________________________________________________________________________________________________________________________




# IMPORTS:
#____________________________________________________________________________________________________________________________________________________
# Basic imports
import os
import re
import textwrap

# RAG components
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LangChain tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# LLM
from langchain_openai import ChatOpenAI

# DO NOT MODIFY THESE
from eval.eval_task1_quant import llm, eval_rag_chain_proj_query

# These are provided as reference
# You do not NEED to use these
from eval.eval_task1_quant import (
    baseline_hfe, baseline_vectorstore,
    baseline_retriever, baseline_rag_chain
)

from parser_SA import *

import json
from langchain_core.documents import Document

# General
# Functions for interacting with the operating system
import os

# Disable parallelism warnings from Hugging Face tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Clean print statements
import textwrap

# Get PDFs as Document object
from langchain_community.document_loaders import PyMuPDFLoader

# Recursive parsing
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Dynamic Semantic Chunking
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker

# For setting up vectorstore
from langchain_chroma import Chroma

# Supports literal conversion
import ast

# Option for easy text cleaning
from textblob import TextBlob

# Allows for async calls to llms
import asyncio

# For comparing serial vs. async calls
import time

# Clients for access TDAC hosted models via OpenAI APIs
from openai import OpenAI, AsyncOpenAI

# To measure semantic similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# For sparse retrieval
from langchain_community.retrievers import BM25Retriever

# For Cross-Encoder retrieval
from sentence_transformers import CrossEncoder

# RAG components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LangChain tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# LLM
from langchain_openai import ChatOpenAI

# Standard library
import json
import logging
import operator
import os
import sys
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from typing import Annotated, List, Literal
from tqdm import tqdm
import random
import re

# Third party
import requests
from IPython.display import Image, display
from typing_extensions import NotRequired, TypedDict

# Pydantic
from pydantic import BaseModel, Field

# OpenAI
from openai import OpenAI

# LangChain Core
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

# LangChain potpurri
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer

# LangChain OpenAI
from langchain_openai import ChatOpenAI

# LangChain Community
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PyMuPDFLoader

# LangGraph
from langgraph.graph import StateGraph, END

# Add the PARENT directory (oa4910) to sys.path and load helper functions
sys.path.append(str(Path.cwd().parent))
from helper_code.rag.load_dataset import setup_embedding_function, load_db_from_dir, load_vectorstore
#____________________________________________________________________________________________________________________________________________________





# DOCUMENT PARSING:
#____________________________________________________________________________________________________________________________________________________
########## PROJECT DATA SHEETS ##########
# Path to documents
directory = "docs/projects"
proj_doc_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pdf')]

# We will build the flat list directly
proj_docs_flat = []

for doc_path in proj_doc_paths:
    try:
        # 1. Parse the PDF using your custom logic
        boxes = FormTextExtractor(doc_path).extract_boxes()
        pds_dict = get_pds(boxes)
        
        # 2. Extract the Project ID (the top-level key) and the data payload
        project_id = list(pds_dict.keys())[0]
        project_data = pds_dict[project_id]
        
        # 3. Convert the nested dictionary into a structured string for the LLM to read.
        page_content = json.dumps(project_data, indent=2)
        
        # 4. Create the LangChain Document with ALL top section fields as metadata
        doc = Document(
            page_content=page_content,
            metadata={
                "source": doc_path,
                "project_id": project_id,
                "title": project_data.get("title"),
                "installation": project_data.get("installation"),
                "CWE": project_data.get("CWE"),
                "CCN": project_data.get("CCN"),
                "region": project_data.get("region"),
                "lead_proponent": project_data.get("lead_proponent"),
                "COCOM": project_data.get("COCOM"),
                "scope": project_data.get("scope"),
                "impact_if_not_provided": project_data.get("impact_if_not_provided")
            }
        )
        
        proj_docs_flat.append(doc)
        
    except Exception as e:
        print(f"Error processing {doc_path}: {type(e).__name__} - {e}")

# Verify
#print(f'Total Documents Processed: {len(proj_docs_flat)}')


########## POM26 STRATEGY ##########
# Path to documents
directory = "docs/strategy/pom26"
strat26_doc_paths = [
    os.path.join(directory, filename) 
    for filename in os.listdir(directory) 
    if filename.endswith('.pdf')
    ]

# Extract text into documents
strat26_docs_multilevel = [
    PyMuPDFLoader(doc_path).load() 
    for doc_path in strat26_doc_paths
    ]

# Flatten docs into 1 dim list
strat26_docs_flat = [
    item 
    for sublist in strat26_docs_multilevel 
    for item in sublist
    ]

########## POM28 STRATEGY ##########
# Path to documents
directory = "docs/strategy/pom28"
strat28_doc_paths = [
    os.path.join(directory, filename) 
    for filename in os.listdir(directory) 
    if filename.endswith('.pdf')
    ]

# Extract text into documents
strat28_docs_multilevel = [
    PyMuPDFLoader(doc_path).load() 
    for doc_path in strat28_doc_paths
    ]

# Flatten docs into 1 dim list
strat28_docs_flat = [
    item 
    for sublist in strat28_docs_multilevel 
    for item in sublist
    ]
#____________________________________________________________________________________________________________________________________________________






# EMBEDDINGS:
#____________________________________________________________________________________________________________________________________________________
# Embedding model
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

baseline_hfe = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
#____________________________________________________________________________________________________________________________________________________






# CHUNKING AND VECTORSTORE CREATION:
#____________________________________________________________________________________________________________________________________________________
# Recursive text splitting
# `chunk_size` and `chunk_overlap` are tunable hyperparameters
# `separators` can also be used to specify what to split on
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

# Creates chunks from documents using the splitter
recursive_proj_docs = recursive_splitter.transform_documents(proj_docs_flat)
recursive_strat26_docs = recursive_splitter.transform_documents(strat26_docs_flat)
recursive_strat28_docs = recursive_splitter.transform_documents(strat28_docs_flat)

# Add unique chunk id as metadata
# Chromadb requires strings for metadata
# Crombadb will throw an error if id is 0
for i, chunk in enumerate(recursive_proj_docs, start=1):
    chunk.metadata['id'] = str(i)
for i, chunk in enumerate(recursive_strat26_docs, start=1):
    chunk.metadata['id'] = str(i)
for i, chunk in enumerate(recursive_strat28_docs, start=1):
    chunk.metadata['id'] = str(i)

# Create vectore store for base recursive method
# Saved locally
recursive_chunk_vectorstore_proj = Chroma.from_documents(
    documents=recursive_proj_docs, 
    embedding=baseline_hfe,
    persist_directory="databases/proj",
    ids=[doc.metadata["id"] for doc in recursive_proj_docs]
)
recursive_chunk_vectorstore_strat26 = Chroma.from_documents(
    documents=recursive_strat26_docs, 
    embedding=baseline_hfe,
    persist_directory="databases/strat26",
    ids=[doc.metadata["id"] for doc in recursive_strat26_docs]
)
recursive_chunk_vectorstore_strat28 = Chroma.from_documents(
    documents=recursive_strat28_docs, 
    embedding=baseline_hfe,
    persist_directory="databases/strat28",
    ids=[doc.metadata["id"] for doc in recursive_strat28_docs]
)

# Load vectorstores:
proj_vectorstore_path: str = "./databases/proj"
proj_vectorstore = load_vectorstore(proj_vectorstore_path, embedding_function=baseline_hfe)

strat26_vectorstore_path: str = "./databases/strat26"
strat26_vectorstore = load_vectorstore(strat26_vectorstore_path, embedding_function=baseline_hfe)

strat28_vectorstore_path: str = "./databases/strat28"
strat28_vectorstore = load_vectorstore(strat28_vectorstore_path, embedding_function=baseline_hfe)
#____________________________________________________________________________________________________________________________________________________






# INITIALIZE LLMS:
#____________________________________________________________________________________________________________________________________________________

API_KEY = "sk-UtrV9i5fFenmG6hvMss71A"
BASE_URL = "http://trac-malenia.ern.nps.edu:8080/inference/v1"

# Check the models available
model_ids = []

try:
    response = requests.get(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {API_KEY}"}
    )
    response.raise_for_status()  
    info = response.json()
    
    for model in info['data']:
        model_ids.append(model['id'])
    model_id = info['data'][0]['id']
    print(f"Available Models: {model_ids}")
    print(f"\nDefault Selected Model Id: {model_id}")
except Exception as e:
    print(f"Error accessing model endpoint: {e}")

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
    match = re.search(r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4})\b', question)
    return match.group(0) if match else None

def semantic_retrieve_w_scores(state: dict) -> dict:
    routes = state.get("routes", [])
    query = state["question"]
    k = state.get("k", 6)
    project_id = extract_project_id(query)
    filter_dict = {"project_id": project_id} if project_id else None

     # Map route names to actual vectorstores
    store_map = {
        "proj_vectorstore": proj_vectorstore,
        "strat26_vectorstore": strat26_vectorstore,
        "strat28_vectorstore": strat28_vectorstore 
    }

    # Build list of actual stores to query
    stores = [store_map[r] for r in routes if r in store_map]

    if not stores:
        return {"documents": []}

    all_docs = []

    for store in stores:
        # Only apply project_id filter to the project vectorstore
        filt = filter_dict if store is proj_vectorstore else None

        docs = store.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filt
        )
        all_docs.extend(docs)

    # Sort combined results by score descending
    all_docs.sort(key=lambda x: x[1], reverse=True)

    logger.info(f"Retrieved {len(all_docs)} documents across {len(stores)} stores")
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
    """
    Post-generate checker.
    Sets state['route_after_check'] to 'websearch' or 'final'
    and increments state['websearch_retries'] when routing to websearch.
    """
    
    logger.info("NODE: Checking if retrieved answer contains IDK")

    generation = state.get("generation", "")
    documents = state.get("documents", []) 

    # Use this entire commented section if you want to preview the retrieved context as part of your output
    # ---- Build context preview ----
    context_preview_parts = []
    for i, (doc, score) in enumerate(documents[:2]):  # limit to first 2 docs
        snippet = (
            doc.page_content
            .replace("\n", " ")
            .strip()[:350]
        )
        context_preview_parts.append(
            f"[doc{i} score={score:.2f}] {snippet}"
        )

    context_preview = " | ".join(context_preview_parts) or "<no documents>"

    # # ---- Generation preview ----
    gen_preview = generation.replace("\n", " ")[:120] if generation else "<empty>"

    # # ---- Side-by-side log ----
    logger.info(
        "\n"
        "  ===== RAG CHECK =====\n"
        f"  Docs count : {len(documents)}\n"
        f"  Context    : {context_preview}\n"
        f"  Generation : {gen_preview}\n"
        "  ====================="
    )
    # end of commented section for previewing retrieved context
    
    #logger.info()

    logger.info("  Generation is acceptable -> final")
    return {
        "route_after_check": "final",
    }

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
                                   "final": END
                               }
                               )

# Compile the graph
app = workflow.compile()

# Display Graph
display(Image(app.get_graph().draw_mermaid_png()))
#____________________________________________________________________________________________________________________________________________________







# EVALUATE:
#____________________________________________________________________________________________________________________________________________________
# Wrap app.invoke to match the expected interface (takes string, returns string)
agent_as_chain = RunnableLambda(
    lambda question: app.invoke({"question": question, "k": 6})["generation"]
)

eval_rag_chain_proj_query(agent_as_chain, q_num=15)

# Test a "multi-part" question. For now, just want to verify that it selects multiple routes:
print("\n\n=== MANUAL MULTI-PART TESTS ===")

test_questions = [
    "What was the justification for project P738, and how should its Mission Alignment score change under the updated POM28 guidance?",
    "Summarize the scope of project RM16-0799 and explain how it aligns with the POM26 National Defense Strategy themes.",
    "Provide the CWE and justification for project NF20-0826, and compare how its Readiness Support score would be interpreted under POM26 versus POM28 criteria.",
    "How do the definitions of Operational Cost differ between POM26 and POM28 scoring guidance?"
]

for q in test_questions:
    print("\n--- QUESTION ---")
    print(q)
    result = app.invoke({"question": q, "k": 6})
    print("\n--- ROUTES SELECTED ---")
    print(result.get("routes", "<no routes>"))
    print("\n--- ANSWER ---")
    print(result["generation"])
    print("\n-------------------------")
#____________________________________________________________________________________________________________________________________________________

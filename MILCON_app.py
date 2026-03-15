# Adapted from code by Luke Holmes

import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI
import pydeck as pdk
# Must 'pip install sqlalchemy' & 'pip install "polars[sqlalchemy]" '
# Must 'pip install connectorx'
# Must 'pip install tabulate'

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

from helper_code.parser import *

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

# Cross-Encoder Re-ranking
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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
# from IPython.display import Image, display
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

# --------------------------------------
# WEBPAGE CONFIGURATION
# --------------------------------------


st.set_page_config(                                     # Streamlit page configuration
    page_title="MILCON and FSRM Project Data Sheet Reference",  # Page title
    layout="wide"                                      # Use wide layout
    # page_icon="✈️"
)

# --------------------------------------
# TITLE & SUBTITLE
# --------------------------------------

st.markdown(                            # Centered dashboard title
    "<h1 style='text-align: center;'>MILCON and FSRM Project Data Sheet (PDS) Reference</h1>",
    unsafe_allow_html=True              # Render HTML safely - not plain text
)

st.markdown(                            # Centered dashboard subtitle
    "<h3 style='text-align: center;'>An LLM-powered tool for exploring MILCON and FSRM project data</h3>",
    unsafe_allow_html=True              # Render HTML safely - not plain text
)

st.markdown("---")                    # Horizontal rule separator


# --------------------------------------
# DESCRIPTION OF DASHBOARD PAGES
# --------------------------------------

st.markdown("""
### Dashboard Overview

This dashboard provides an AI-powered interface for exploring **MILCON** (Military Construction) 
and **FSRM** (Facilities Sustainment, Restoration, and Modernization) project data for the 
**CNIC Navy Region Europe, Africa, and Southwest Asia (CNR EU)**.

---

#### 📁 Data Sources
The assistant draws from three document domains:

| Domain | Description |
|---|---|
| **Project Data Sheets (PDS)** | Individual project datasheets including scope, cost (CWE), location, facility details, and scoring justifications |
| **POM26 Strategic Guidance** | NSS/NDS themes, CNIC scoring criteria, and PDS scoring definitions as established during POM26 |
| **POM28 Strategic Guidance** | Updated NSS/NDS themes, revised CNIC scoring criteria, and new scoring definitions for POM28 |

---

#### 💬 What You Can Ask
The assistant supports two types of queries:

**Project-Specific Questions**
- Retrieve scope, cost, location, or facility details for a specific project
- Report existing mission alignment, readiness support, operational cost, severity, and urgency scores
- Explain how a project aligns with POM26 strategic guidance
- Estimate how a project *would* score under updated POM28 guidance

**Dataset Summary Questions**
- Count or list projects by country, installation, or region
- Filter projects by score thresholds (e.g., mission alignment ≥ 4)
- Calculate total or average costs across countries or regions
- Compare project portfolios across multiple locations

---

#### 🔍 Example Questions
- *"List all projects in Italy with a region mission alignment score of 4"*
- *"What is the total cost of all projects in Greece and Spain?"*
- *"What is the scope and impact of P309?"*
- *"How does P740 align with POM26 guidance?"*
- *"How would RM23-0514 score under POM28 strategic guidance?"*
- *"How many projects have a readiness support score below 3?"*

""", unsafe_allow_html=True)

# --------------------------------------
# LLM HELPER FUNCTIONS
# --------------------------------------

# After the user inputs the filter parameters (i.e. start/end year, aircraft, etc.), 
# under the hood the dashboard has a Polars df with the returned query dat. We can give 
# the LLM summary stats or small samples of this df to help it answer questions. We can 
# feed some of this df (i.e. small samples of rows to show patterns, summary stats, etc.)
# plus the user's questions to the LLM as context to the model to produce better answers.
def is_llm_configured():
    """Check whether Streamlit sees your NVIDIA API key"""    
    return "NVIDIA_API_KEY" in st.secrets

# --------------------------------------
# VECTORSTORE & CLIENT SETUP
# --------------------------------------

system_prompt = (
    "You are a helpful assistant for MILCON and FSRM project data. "
    "Answer questions using only the retrieved project and strategy documents."
)

@st.cache_resource
def load_resources():
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    proj_vs = load_vectorstore("./databases/proj", embedding_function=embedding_function)
    strat26_vs = load_vectorstore("./databases/strat26", embedding_function=embedding_function)
    strat28_vs = load_vectorstore("./databases/strat28", embedding_function=embedding_function)
    return embedding_function, proj_vs, strat26_vs, strat28_vs


@st.cache_resource
def load_project_dataframe(_proj_vectorstore):
    """Pull all chunks from Chroma and merge them per project_id, including scores from page_content."""
    collection = _proj_vectorstore._collection
    results = collection.get(include=["documents", "metadatas"])
    
    from collections import defaultdict
    project_chunks = defaultdict(list)
    
    for doc_str, meta in zip(results["documents"], results["metadatas"]):
        pid = meta.get("project_id")
        if not pid:
            continue
        
        # Start with metadata fields
        merged_chunk = dict(meta)
        
        # Parse the page_content JSON to get ALL fields including scores
        try:
            parsed = json.loads(doc_str)
            if isinstance(parsed, dict):
                for key, val in parsed.items():
                    # Only add if not already in metadata or if richer value
                    if key not in merged_chunk or merged_chunk[key] is None:
                        merged_chunk[key] = val
        except (json.JSONDecodeError, TypeError):
            pass
        
        project_chunks[pid].append(merged_chunk)
    
    # Merge all chunks per project — fill in missing fields from later chunks
    rows = []
    for pid, chunks in project_chunks.items():
        merged = {}
        for chunk in chunks:
            for key, val in chunk.items():
                if key not in merged or merged[key] is None or merged[key] == "":
                    merged[key] = val
        rows.append(merged)
    
    df = pd.DataFrame(rows)
    print(f"[DataFrame] Built {len(df)} unique projects from {sum(len(v) for v in project_chunks.values())} chunks")
    print(f"[DEBUG] proj_df columns: {list(df.columns)}")
    
    # Show scoring columns specifically
    score_cols = [c for c in df.columns if any(k in c.lower() for k in 
                  ['mission', 'severity', 'readiness', 'operational', 'score', 'alignment'])]
    print(f"[DEBUG] Scoring columns found: {score_cols}")
    print(f"[DEBUG] sample row keys: {list(df.iloc[0].to_dict().keys()) if len(df) > 0 else 'empty'}")
    
    return df

# Call it right after load_resources()
baseline_hfe, proj_vectorstore, strat26_vectorstore, strat28_vectorstore = load_resources()
proj_df = load_project_dataframe(proj_vectorstore) 
print("[DEBUG] proj_df columns:", list(proj_df.columns))
print("[DEBUG] sample row:", proj_df.iloc[0].to_dict())

@st.cache_resource
def build_llm_clients():
            ########## ROUTER MODEL ##########
    llm_md_tools = ChatOpenAI(
        base_url=st.secrets["NVIDIA_API_BASE"],
        model=st.secrets["NVIDIA_MODEL"],
        api_key=st.secrets["NVIDIA_API_KEY"],
        temperature=0,  
        name= "qwen" # Name the LLM for langchain
    )

    ########## GENERATION MODEL ##########
    llm_gen_tools = ChatOpenAI(
        base_url=st.secrets["NVIDIA_API_BASE"],
        model= st.secrets["NVIDIA_MODEL"],
        api_key=st.secrets["NVIDIA_API_KEY"],
        temperature=0.3,
        name="qwen_gen"
    )

    ########## GRADER MODEL ##########
    llm_grader = ChatOpenAI(
        base_url=st.secrets["NVIDIA_API_BASE"],
        model=st.secrets["NVIDIA_MODEL"],
        api_key=st.secrets["NVIDIA_API_KEY"],
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
    return llm_md_tools, llm_gen_tools, grader_chain

llm_md_tools, llm_gen_tools, grader_chain = build_llm_clients()

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

    # Add this temporarily at the top of run_llm_intro, before anything else
# from openai import OpenAI
# client = OpenAI(
#     base_url=st.secrets["NVIDIA_API_BASE"],
#     api_key=st.secrets["NVIDIA_API_KEY"]
# )
# models = client.models.list()
# for m in models.data[:-1]:
#     logger.info(f"  AVAILABLE MODEL: {m.id}")
    

logger.info("Logger configured for tool creation and testing")

def run_llm_intro(user_msg: str, history: list):
    """Call NVIDIA's hosted model we set in secrets.toml
       with some context from the currently filtered dataframe.
       This function is for the INTRO PAGE - no df is passed in.
    --------------------------------------
    Args:
        user_msg (str): user's inputs into the chat
        history (list[dicts[str]]): record of the user's chat messages for context
    --------------------------------------
    Returns:
        string: LLM's response to the user's inputs (comes from a ChatCompletion object)
    """    
    AGGREGATE_KEYWORDS = [
        "how many",
        "count",
        "total cost",
        "total amount",
        "total cwe",
        "combined cost",
        "sum of",
        "list all",
        "show all",
        "all projects",
        "which projects",
        "average cost",
        "avg cost",
        "projects in",       # catches "projects in greece and italy"
        "projects across",
    ]

    # Phrases that should NEVER go to pandas even if they contain aggregate keywords
    SPECIFIC_PROJECT_PATTERNS = [
        r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4}|B\d{3,4})\b',  # specific project ID
    ]

    NARRATIVE_KEYWORDS = [
        "justification",
        "justify",
        "explain",
        "why",
        "how does",
        "align with",
        "strategy",
        "strategic",
        "guidance",
        "scope",
        "impact",
        "description",
        "what is the",
        "tell me about",
        "summarize",
    ]

    def is_aggregate_query(question: str) -> bool:
        """Return True only for dataset-wide summary/count/filter queries."""
        # Extract just the current question (strip conversation history)
        if "Current question:" in question:
            q = question.split("Current question:")[-1].strip().lower()
        else:
            q = question.lower()

        # Never aggregate if asking about a specific project ID
        for pattern in SPECIFIC_PROJECT_PATTERNS:
            if re.search(pattern, q, re.IGNORECASE):
                return False

        # Never aggregate if question is narrative/explanatory
        if any(kw in q for kw in NARRATIVE_KEYWORDS):
            return False

        # Only aggregate if a clear summary keyword is present
        return any(kw in q for kw in AGGREGATE_KEYWORDS)

    # create logging runnable to track router chain execution
    def logging_helper(state: dict) -> dict:
        """Log internal chain steps"""
        logger.debug(f"Intermediary State: '{state}'")
        return state

    r_logger = RunnableLambda( logging_helper, name= "log chain state")

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

        full_question = state["question"]

        # Extract just the current question for routing
        if "Current question:" in full_question:
            current_q = full_question.split("Current question:")[-1].strip()
        else:
            current_q = full_question

        # Detect POM28 re-score/update requests and inject context for router
        POM28_KEYWORDS = [
            "pom28", "pom 28", "update this", "rewrite this", "update description",
            "under pom28", "using pom28", "new guidance", "updated guidance",
            "re-score", "rescore", "estimate under"
        ]
        POM26_KEYWORDS = [
            "pom26", "pom 26", "existing guidance", "original guidance",
            "as scored", "current score"
        ]

        # Build an enriched routing query that includes relevant context
        if any(kw in current_q.lower() for kw in POM28_KEYWORDS):
            # Force include strat28 by appending context to routing query
            routing_query = f"{current_q}\n\n[NOTE: This question requires POM28 strategy guidance to answer.]"
            logger.info("  Detected POM28 update/rescore request — injecting strat28 hint")
        elif any(kw in current_q.lower() for kw in POM26_KEYWORDS):
            routing_query = f"{current_q}\n\n[NOTE: This question requires POM26 strategy guidance to answer.]"
            logger.info("  Detected POM26 alignment request — injecting strat26 hint")
        else:
            routing_query = current_q

        logger.debug(f"  Question to route: {routing_query}")

        result = router_chain.invoke({"query": routing_query})
        routes = result.tool_calls[0]["args"]["routes"]

        # Safety net: if current question has POM28 keywords but router missed strat28, add it
        if any(kw in current_q.lower() for kw in POM28_KEYWORDS):
            if "strat28_vectorstore" not in routes:
                routes.append("strat28_vectorstore")
                logger.info("  Safety net: added strat28_vectorstore to routes")
        if any(kw in current_q.lower() for kw in POM26_KEYWORDS):
            if "strat26_vectorstore" not in routes:
                routes.append("strat26_vectorstore")
                logger.info("  Safety net: added strat26_vectorstore to routes")

        # Also ensure proj_vectorstore is always included when a project ID is in history
        # Only skip history-based proj_vectorstore injection if:
        # 1. The current question is about strategy/policy AND
        # 2. There is NO specific project ID in the current question itself
        current_project_id = extract_project_id(current_q)

        PURE_STRATEGY_KEYWORDS = [
            "national security strategy", "nss", "nds", "national defense strategy",
            "key focus areas", "strategic themes", "pom26 strategy", "pom28 strategy",
            "cnic scoring", "scoring criteria", "scoring definitions"
        ]

        is_pure_strategy = (
            any(kw in current_q.lower() for kw in PURE_STRATEGY_KEYWORDS)
            and current_project_id is None  # only pure strategy if no project ID present
        )

        if not is_pure_strategy:
            project_id = extract_project_id_from_history(full_question)
            if project_id and "proj_vectorstore" not in routes:
                routes.append("proj_vectorstore")
                logger.info(f"  Safety net: added proj_vectorstore for project {project_id}")

        logger.info(f"  Routing decision: {routes}")
        return {"routes": routes}

    def pandas_query_node(state: dict) -> dict:
        logger.info("NODE: Pandas Aggregate Query")
        question = state["question"]

        # Pre-defined helper injected into exec scope
        def get_score(x):
            if isinstance(x, dict):
                return x.get("score")
            try:
                v = float(x)
                return int(v) if v == int(v) else v
            except (TypeError, ValueError):
                return None

        # Find all scoring columns that actually exist
        score_cols = [c for c in proj_df.columns if "mission" in c.lower()
                    or "alignment" in c.lower()
                    or "severity" in c.lower()
                    or "readiness" in c.lower()
                    or "operational" in c.lower()
                    or "score" in c.lower()]

        schema_info = f"""
        You have a pandas DataFrame called `proj_df` with {len(proj_df)} rows.

        ACTUAL columns: {list(proj_df.columns)}

        SCORING columns (all plain integers, -1 means not provided):
        {score_cols}

        KEY COLUMN NOTES:
        - 'installation': ends in country code (' IT'=Italy, ' GR'=Greece, ' SP'=Spain, ' JA'=Japan, ' BH'=Bahrain)
        - 'CWE': cost in THOUSANDS of dollars (100000 = $100M)
        - SCORING COLUMNS ARE PLAIN INTEGERS — filter directly: proj_df['region_mission_alignment_score'] >= 3
        - A value of -1 means score not provided — exclude with col > 0
        - Do NOT define any functions — use direct pandas operations or lambdas only
        - IMPORTANT: str.endswith() does NOT accept a list — use a tuple instead:
        CORRECT:   proj_df['installation'].str.endswith((' IT', ' GR'))
        INCORRECT: proj_df['installation'].str.endswith([' IT', ' GR'])
        - For multiple countries, always use a tuple in str.endswith()
        - 'project_id': string like 'P309', 'RM23-0509'
        """

        code_prompt = f"""
    {schema_info}

    Write Python/pandas code to answer this question: "{question}"

    RULES:
    - Store the final answer in a variable called `result`
    - `result` must be a string — format it nicely using markdown
    - For lists of projects, format as a markdown table with columns: | Project ID | Title | Installation |
    - For simple counts or single values, return a plain string like "There are 5 projects in Italy."
    - For mixed answers (count + list), combine: start with the count sentence, then the table
    - Handle None/NaN safely with pd.notna() or isinstance() checks
    - Do NOT import anything — pandas is available as `pd`
    - Do NOT define any functions — use direct pandas operations or lambdas only
    - Output ONLY the code, no explanation, no markdown fences

    MARKDOWN TABLE FORMAT EXAMPLE:
    result = "There are 3 projects in Italy.\\n\\n| Project ID | Title | Installation |\\n|---|---|---|\\n"
    for _, row in filtered_df.iterrows():
        result += f"| {{row['project_id']}} | {{row.get('title', 'N/A')}} | {{row.get('installation', 'N/A')}} |\\n"
    """

        response = llm_gen_tools.invoke([{"role": "user", "content": code_prompt}])
        code = response.content.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()

        logger.debug(f"  Generated pandas code:\n{code}")

        # Inject get_score helper so it's in scope even if LLM uses it anyway
        local_vars = {"proj_df": proj_df.copy(), "pd": pd, "json": json, "get_score": get_score}

        try:
            exec(code, {}, local_vars)
            result_str = str(local_vars.get("result", "Could not compute result."))
        except KeyError as e:
            score_cols = [c for c in proj_df.columns if "score" in c.lower()
                        or "alignment" in c.lower() or "severity" in c.lower()]
            result_str = (
                f"Could not find the column {e} in the dataset. "
                f"Available scoring-related columns are: {score_cols}. "
                f"Try rephrasing your question using one of these field names."
            )
            logger.error(f"  KeyError: {e}\nCode:\n{code}")
        except Exception as e:
            result_str = f"Error running that query: `{e}`"
            logger.error(f"  Pandas exec error: {e}\nCode:\n{code}")

        logger.info(f"  Pandas result: {result_str[:200]}")
        return {"generation": result_str}


    # Build the retrieval node
    def extract_project_id(question: str) -> str | None:
        match = re.search(r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4}|B\d{3,4})\b', question, re.IGNORECASE)
        return match.group(0).upper() if match else None

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

    GROUNDING RULES
    - Treat the retrieved documents as the authoritative source for all project, facility, and strategy content.
    - Do not invent or infer information that is not supported by the documents or tool outputs.
    - Documents are JSON-formatted and contain fields like installation, title, region, scope, CWE (cost), COCOM, and scoring fields.
    - Project IDs like "P314", "RM16-0799" refer to the "project_id" field. Normalize to uppercase.
    - The "installation" field is the physical location. Infer city/country if recognizable.
    - Scoring fields come in pairs: "region_<category>" and "lead_proponent_<category>". Each has "score" and "description" subfields. Null = not provided.

    MISSING-EVIDENCE RULES
    - If no documents are retrieved and the question concerns Navy facilities projects, respond with "I don't know. No relevant project documents were retrieved to answer this question. Are you sure you have the correct project ID?"
    - If no documents are retrieved and the question is general, you may answer using general knowledge.

    ANSWERING RULES - EXISTING SCORES (POM26 / as-scored):
    - If the user asks how a project aligns with POM26, or asks for existing scores/justifications,
    report the scores and descriptions EXACTLY as stored in the document fields.
    - Always report BOTH region and lead_proponent scores. If either is null, state "not provided".
    - Cite the specific field you are drawing from (e.g., "According to region_mission_alignment_desc...").
    - Format scores clearly: "Region Mission Alignment: 4 — [description]"

    ANSWERING RULES - NEW SCORING UNDER POM28 (re-score requests):
    - If the user asks how a project WOULD align, COULD score, or asks to RE-SCORE under POM28 guidance,
    you are being asked to GENERATE a new estimated score and justification.
    - Use the POM28 strategy documents retrieved to understand the updated NSS/NDS themes and scoring criteria.
    - Use the project's scope, mission, and impact fields as the basis for the new justification.
    - Clearly label your output as an ESTIMATED score under POM28, not the official recorded score.
    - Format as: "**Estimated POM28 Score: [1-5]** — [justification referencing both the project details and POM28 themes]"
    - If POM28 docs were not retrieved, say you cannot estimate without the POM28 guidance documents.

    GENERAL RULES
    - Answer in 3-6 sentences for alignment questions — these require more detail than simple factual lookups.
    - if asked about cost or CWE, convert to dollars (e.g. CWE 68140 = $68,140,000).
    - When POM26 and POM28 guidance conflict, clearly distinguish between them.
    - Never say "I don't know" if the documents contain any relevant information.
    """),
            ("human", """Retrieved Documents:
    {context}

    Question: {query}
            
            {answer_instructions}""")
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

        context_parts = []
        for doc, score in documents:
            project_id = doc.metadata.get("project_id", "")
            installation = doc.metadata.get("installation", "")
            header = ""
            if project_id:
                header = f'[project_id: {project_id}, installation: {installation}]\n'
            context_parts.append(header + doc.page_content)

        context = "\n\n---\n\n".join(context_parts)

        has_choices = bool(re.search(r'\b[A-D]\)', question))
        
        if has_choices:
            answer_instructions = (
                "Instructions: You MUST respond with ONLY one of the provided answer choices "
                "exactly as written (e.g. 'D) 4'). Do NOT make up an answer that is not one "
                "of the options. Do not include any explanation or additional text."
            )
        else:
            answer_instructions = (
                "Instructions: Answer the question clearly and concisely using the provided documents. "
                "Use any relevant fields in the documents including installation, region, scope, title, "
                "and other metadata to answer the question. Only say \"I don't know\" if the documents "
                "contain absolutely no relevant information. Simply provide the answer without any additional commentary or explanation. Be direct and to the point."
            )

        logger.debug(f"  Context preview: {context[:100]}...")
        logger.debug(f"  Context length: {len(context)} chars")
        
        # # Create the prompt from state
        logger.debug(f"  Full context:\n{context}")
        prompt = rag_template.invoke({"query": question, "context": context, "answer_instructions": answer_instructions})
        logger.info(f"  Prompt created with {len(prompt.messages)} messages")
        logger.debug(f"  Formatted prompt messages: {[m.type for m in prompt.messages]}")
        
        # Invoke LLM and generate
        logger.info("  Invoking LLM...")
        # Add this right before msg = llm_gen_tools.invoke(prompt.messages)
        for m in prompt.messages:
            logger.debug(f"  PROMPT MSG [{m.type}]:\n{m.content}")
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

        # Extract current question only (strip conversation history)
        if "Current question:" in question:
            current_q = question.split("Current question:")[-1].strip().lower()
        else:
            current_q = question.lower()

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

        # ---- Skip grader for generative/alignment questions ----
        # These require synthesis across documents — grader will incorrectly reject
        # valid answers because generated text won't match retrieved chunks verbatim
        GENERATIVE_KEYWORDS = [
            "align", "alignment", "strategy", "strategic", "pom26", "pom28",
            "would", "could", "estimate", "re-score", "rescore", "justify",
            "justification", "how does", "how would", "explain"
        ]
        if any(kw in current_q for kw in GENERATIVE_KEYWORDS):
            logger.info("  Generative/alignment question detected — skipping grader, returning final")
            return {"route_after_check": "final"}

        # ---- LLM grounding check for open-ended factual questions ----
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
        question = state.get("question", "")
        if is_aggregate_query(question):
            logger.info("EDGE: route_question_edge -> pandas_aggregate")
            return "pandas_aggregate"
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
    workflow.add_node("pandas_aggregate", pandas_query_node)  
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
                        "pandas_aggregate": "pandas_aggregate",
                        "retrieve": "semantic_retriever_and_grader" 
                    }
                    )

    # retrieve -> generate
    workflow.add_edge("pandas_aggregate", END)  
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

    # Build context-aware question from history
    if history:
        history_text = "\n".join([
            f"{m['role'].upper()}: {m['content']}" 
            for m in history[-4:]  # last 4 messages
        ])
        enriched_question = f"Conversation so far:\n{history_text}\n\nCurrent question: {user_msg}"
    else:
        enriched_question = user_msg

    result = app.invoke({"question": enriched_question})

    # At the bottom of run_llm_intro, replace the return statement:
    generation = result.get("generation", "I don't know.")
    generation = re.sub(r'<think>.*?</think>', '', generation, flags=re.DOTALL).strip()
    documents  = result.get("documents", [])

    return generation, documents  # <-- return both

# --------------------------------------
# CHAT ASSISTANT UI
# --------------------------------------
st.markdown("---")                                  # Horizontal grey line to separate sections
st.subheader("Ask the dashboard assistant...")      # Chat assistant subheader

# Fail gracefully if the LLM isn't configured
if not is_llm_configured():                 # If the LLM isn't configured
    st.info(                                # Show info message - point the user to the README
        "LLM assistant not configured. "
        "If you're running this app yourself, add your NVIDIA API key to "
        "`.streamlit/secrets.toml` (see README & Tutorial_Dashboard.md for instructions)."
    )
else:                                       # If the LLM is configured
    if "messages" not in st.session_state:  # Initialize chat history in session state
        st.session_state.messages = []      # Empty list to hold messages - each message is a dict with "role" & "content" keys

    for msg in st.session_state.messages:   # Display all prior messages in the chat history
        with st.chat_message(msg["role"]):  # Role is either "user" or "assistant"
            st.markdown(msg["content"])     # Content is the message text - display with markdown formatting

    # Chat input
    user_msg = st.chat_input(               # Input box for user to type messages
        "Ask me a question about a MILCON or FSRM project (only NAVEUR/NAVAF for now!)"
    )

    if user_msg:                                    # If the user submitted a message
        st.session_state.messages.append(           # Save user message to history
            {"role": "user", "content": user_msg}   # Remember, it's the 'user' role
        )                                           

        with st.chat_message("user"):               # Display the user's message in the chat
            st.markdown(user_msg)

        # Here's the critical part - call the LLM API with the user's message, 
        # the filtered df, & the prior chat history for context
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    assistant_reply, source_docs = run_llm_intro(
                        user_msg=user_msg,
                        history=st.session_state.messages[:-1]
                    )
                except Exception as e:
                    assistant_reply = f"Sorry, I couldn't reach the LLM API: `{e}`"
                    source_docs = []

                st.markdown(assistant_reply)

                # Show sources in collapsible expander
                if source_docs:
                    with st.expander("📄 Sources", expanded=False):
                        seen = set()
                        for doc, score in source_docs:
                            pid = doc.metadata.get("project_id")
                            title = doc.metadata.get("title", "")
                            install = doc.metadata.get("installation", "")
                            source_file = doc.metadata.get("source", "")
                            filename = source_file.split("/")[-1] if source_file else "Unknown"

                            # Use filename as identifier for strategy docs (no project_id)
                            dedup_key = pid if pid else filename
                            if dedup_key in seen:
                                continue
                            seen.add(dedup_key)

                            if pid:
                                st.markdown(f"**{pid}** — {title}  \n`{install}` | File: `{filename}`")
                            else:
                                # Strategy document — show filename as the reference
                                st.markdown(f"📘 **{filename}**")

        # Save assistant reply to history
        st.session_state.messages.append(
            {"role": "assistant", 
             "content": assistant_reply}
        )
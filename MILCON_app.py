# Adapted from code by Luke Holmes

import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI
import pydeck as pdk

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
from eval.eval_task1_quant import (
    baseline_hfe, baseline_vectorstore,
    baseline_retriever, baseline_rag_chain
)

from helper_code.parser import *

import json
from langchain_core.documents import Document

# General
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import textwrap

# Document loaders & splitters
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma

import ast
from textblob import TextBlob
import asyncio
import time

from openai import OpenAI, AsyncOpenAI

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

import json
import logging
import operator
import os
import sys
from pathlib import Path
from pprint import pprint
from types import SimpleNamespace
from typing import Annotated, List, Literal, Optional
from tqdm import tqdm
import random
import re

import requests
from typing_extensions import NotRequired, TypedDict
from pydantic import BaseModel, Field
from openai import OpenAI

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from chromadb.utils.embedding_functions import HuggingFaceEmbeddingServer
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.document_loaders import PyMuPDFLoader
from langgraph.graph import StateGraph, END

sys.path.append(str(Path.cwd().parent))
from helper_code.rag.load_dataset import setup_embedding_function, load_db_from_dir, load_vectorstore

# --------------------------------------
# WEBPAGE CONFIGURATION
# --------------------------------------

st.set_page_config(
    page_title="MILCON and FSRM Project Data Sheet Reference",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align: center;'>MILCON and FSRM Project Data Sheet (PDS) Reference</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align: center;'>An LLM-powered tool for exploring MILCON and FSRM project data</h3>",
    unsafe_allow_html=True
)

st.markdown("---")

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
| **POM26 Strategic Guidance** | National Security Strategy (NSS), National Defense Strategy (NDS), CNIC scoring criteria, and PDS scoring definitions as established during POM26 |
| **POM28 Strategic Guidance** | Updated NSS/NDS themes |

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

def is_llm_configured():
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
    collection = _proj_vectorstore._collection
    results = collection.get(include=["documents", "metadatas"])
    
    from collections import defaultdict
    project_chunks = defaultdict(list)
    
    for doc_str, meta in zip(results["documents"], results["metadatas"]):
        pid = meta.get("project_id")
        if not pid:
            continue
        merged_chunk = dict(meta)
        try:
            parsed = json.loads(doc_str)
            if isinstance(parsed, dict):
                for key, val in parsed.items():
                    if key not in merged_chunk or merged_chunk[key] is None:
                        merged_chunk[key] = val
        except (json.JSONDecodeError, TypeError):
            pass
        project_chunks[pid].append(merged_chunk)
    
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
    
    score_cols = [c for c in df.columns if any(k in c.lower() for k in 
                  ['mission', 'severity', 'readiness', 'operational', 'score', 'alignment'])]
    print(f"[DEBUG] Scoring columns found: {score_cols}")
    print(f"[DEBUG] sample row keys: {list(df.iloc[0].to_dict().keys()) if len(df) > 0 else 'empty'}")
    
    return df

baseline_hfe, proj_vectorstore, strat26_vectorstore, strat28_vectorstore = load_resources()
proj_df = load_project_dataframe(proj_vectorstore)
print("[DEBUG] proj_df columns:", list(proj_df.columns))
print("[DEBUG] sample row:", proj_df.iloc[0].to_dict())

@st.cache_resource
def build_llm_clients():
    llm_md_tools = ChatOpenAI(
        base_url=st.secrets["NVIDIA_API_BASE"],
        model=st.secrets["NVIDIA_MODEL"],
        api_key=st.secrets["NVIDIA_API_KEY"],
        temperature=0,
        name="qwen"
    )

    llm_gen_tools = ChatOpenAI(
        base_url=st.secrets["NVIDIA_API_BASE"],
        model=st.secrets["NVIDIA_MODEL"],
        api_key=st.secrets["NVIDIA_API_KEY"],
        temperature=0.3,
        name="qwen_gen"
    )

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

logger = logging.getLogger("agentic_workflow")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

logger.info("Logger configured for tool creation and testing")


def run_llm_intro(user_msg: str, history: list):

    def logging_helper(state: dict) -> dict:
        logger.debug(f"Intermediary State: '{state}'")
        return state

    r_logger = RunnableLambda(logging_helper, name="log chain state")

    # -----------------------------------------------------------------------
    # OPTION 1: UNIFIED ROUTER SCHEMA
    # The router LLM now decides BOTH which stores to query AND how to answer.
    # This replaces all keyword lists for aggregate detection, NSS/NDS rewriting,
    # and POM26/POM28 context detection.
    # -----------------------------------------------------------------------

    class RouteSelection(BaseModel):
        """Unified routing decision returned by the router LLM.

        Fields
        ------
        routes : list of vectorstores to query
        answer_strategy : how the graph should answer the question
            - "pandas_aggregate"  → run pandas code against proj_df
            - "semantic_rag"      → retrieve docs and generate a narrative answer
        retrieval_query : the rewritten search query to use for vector retrieval.
            The router rewrites the user's question into an optimal semantic search
            string (e.g. expanding abbreviations, adding POM-era context).
            Only used when answer_strategy == "semantic_rag".
        reasoning : short explanation of the routing decision (for debugging)
        """
        routes: List[Literal["proj_vectorstore", "strat26_vectorstore", "strat28_vectorstore"]]
        answer_strategy: Literal["pandas_aggregate", "pandas_list", "semantic_rag"]
        retrieval_query: str
        reasoning: str

    @tool(args_schema=RouteSelection)
    def select_route(
        routes: list[str],
        answer_strategy: str,
        retrieval_query: str,
        reasoning: str,
    ) -> dict:
        """Return the full routing decision."""
        return {
            "routes": routes,
            "answer_strategy": answer_strategy,
            "retrieval_query": retrieval_query,
            "reasoning": reasoning,
        }

    llm_router = llm_md_tools.bind_tools([select_route], tool_choice="select_route")

    # -----------------------------------------------------------------------
    # ROUTER PROMPT
    # All routing logic lives here — no keyword lists in Python code.
    # -----------------------------------------------------------------------

    router_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a routing assistant for a Navy MILCON/FSRM project Q&A system.
Your job is to analyse the user's question and return a routing decision with four fields.

────────────────────────────────────────────────────────────
FIELD 1 — routes  (which vectorstores to query)
────────────────────────────────────────────────────────────
Choose ALL that apply from:
  • proj_vectorstore    — project datasheets: scope, cost (CWE), location, facility details,
                          scoring (Mission Alignment, Readiness Support, Operational Cost,
                          Severity, Urgency), proponent info.
  • strat26_vectorstore — POM26-era strategic guidance: NSS/NDS themes, CNIC scoring
                          criteria and definitions as they existed during POM26.
  • strat28_vectorstore — POM28 strategic guidance: updated NSS/NDS themes and revised
                          CNIC scoring criteria used for re-scoring projects.

Rules:
  - Include proj_vectorstore for any question about a specific project ID (e.g. P738,
    RM16-0799), project attributes, scoring, justification, or facility details.
  - Include strat26_vectorstore for questions about POM26 strategy, "existing guidance",
    or "original" scoring.
  - Include strat28_vectorstore for questions about POM28 strategy, "updated guidance",
    "new guidance", re-scoring, or estimating scores under updated policy.
  - If a question asks how a project aligns with strategy, include BOTH proj_vectorstore
    AND the relevant strategy store(s).
  - For aggregate/filter questions (counts, totals, lists), include proj_vectorstore only
    unless strategy context is also explicitly requested.

────────────────────────────────────────────────────────────
FIELD 2 — answer_strategy  (how to answer)
────────────────────────────────────────────────────────────
Choose exactly one:
  • "pandas_aggregate"  — use this when the question asks for:
        - Counts, totals, sums, or averages ACROSS multiple projects
          (e.g. "how many projects in Italy", "total cost of all projects in Greece and Spain")
        - Lists or rankings of projects matching a CONCRETE filter
          (e.g. "list all projects with mission alignment score of 4",
                "which projects are in Spain?", "show all projects under CCN 14380",
                "list all Italy projects with their prices")
        - Dataset-wide comparisons or summaries based on existing data columns
        Do NOT use this if the question is about a single named project.
        Do NOT use this if the question requires qualitative reasoning or interpretation.

  • "pandas_list"       — use this when the question asks for a list of projects AND
        the answer requires filtering on existing data columns PLUS a brief explanation
        per project drawn from the data. Examples:
          - "which projects have a low readiness score and why"
          - "what projects have the weakest justification" (filter low scores, explain desc)
          - "list projects in Italy with poor severity scores and explain"
        The output should be a bulleted list with a brief explanation per project.
        Do NOT use this for simple column-filter queries — those are pandas_aggregate.
        Do NOT use this if answering requires reasoning about strategy or policy
        (e.g. "what projects would score worse under POM28") — that is semantic_rag.
        Do NOT use this if the question is about a single named project.

  • "semantic_rag"      — use this for everything else:
        - Narrative questions about a specific project (scope, justification, impact)
        - Strategy/policy alignment questions
        - Re-scoring or estimating scores under POM28
        - Questions that require reasoning about how strategic themes affect scoring,
          even across multiple projects — e.g. "what projects would score worse under
          POM28 guidance", "which project types benefit most from POM28 themes"
        - "How does P740 align with...", "What is the scope of RM23-0514", etc.

────────────────────────────────────────────────────────────
FIELD 3 — retrieval_query  (optimised search string)
────────────────────────────────────────────────────────────
Rewrite the user's question into the best possible semantic search query.
Rules:
  - Expand abbreviations: "NSS" → "national security strategy", "NDS" → "national defense strategy"
  - Add POM-era context when relevant:
      POM26 NSS/NDS  → "national security strategy priorities competition democracy alliances deterrence"
      POM28 NSS/NDS  → "2025 national security strategy america first priorities strength deterrence"
      POM26 NDS      → "national defense strategy military deterrence threats integrated"
      POM28 NDS      → "2026 national defense strategy military strength deterrence homeland"
  - For project-specific questions, include the project ID and key concepts from the question
  - For aggregate questions, this field is less critical — echo the user's question

────────────────────────────────────────────────────────────
FIELD 4 — reasoning  (one sentence explaining your decision)
────────────────────────────────────────────────────────────
Briefly explain why you chose these routes and this answer_strategy.
This is used only for debugging.
"""),
        ("human", "{query}")
    ])

    router_chain = router_prompt | r_logger | llm_router

    # -----------------------------------------------------------------------
    # GRAPH STATE
    # -----------------------------------------------------------------------

    class GraphState(TypedDict):
        question: str
        answer_strategy: NotRequired[str]
        routes: NotRequired[list[str]]
        retrieval_query: NotRequired[str]
        generation: NotRequired[str]
        max_retries: NotRequired[int]
        gen_attempts: NotRequired[Annotated[int, operator.add]]
        documents: NotRequired[list[tuple[Document, int]]]
        k: NotRequired[int]
        route_after_check: NotRequired[str]
        reasoning: NotRequired[str]

    # -----------------------------------------------------------------------
    # NODES
    # -----------------------------------------------------------------------

    def init_graph_state(state: dict) -> GraphState:
        logger.info("NODE: Initialize State")
        logger.debug(f"  Input state: {state}")
        initialized: GraphState = {
            "question":        state["question"],
            "answer_strategy": state.get("answer_strategy", "semantic_rag"),
            "routes":          state.get("routes", []),
            "retrieval_query": state.get("retrieval_query", state["question"]),
            "generation":      state.get("generation", ""),
            "max_retries":     state.get("max_retries", 3),
            "gen_attempts":    state.get("gen_attempts", 0),
            "documents":       state.get("documents", []),
            "k":               state.get("k", 3),
            "reasoning":       state.get("reasoning", ""),
        }
        logger.debug(f"  Initialized state keys: {list(initialized.keys())}")
        return initialized

    # ------------------------------------------------------------------
    # Route question node — now simply calls the router LLM and stores
    # all four fields in state. No keyword lists anywhere.
    # ------------------------------------------------------------------

    def route_question_node(state: dict) -> dict:
        logger.info("NODE: Route Question")

        full_question = state["question"]

        # Strip conversation history so the router only sees the current question
        # Extract current question
        if "Current question:" in full_question:
            current_q = full_question.split("Current question:")[-1].strip()
        else:
            current_q = full_question

        # Build a context-aware routing query that includes the last assistant turn
        # so the router understands follow-up intent
        if "ASSISTANT:" in full_question:
            last_assistant = full_question.split("ASSISTANT:")[-1].split("Current question:")[0].strip()
            # Summarize to first line only — we don't want to flood the router with a 32-row table
            last_assistant_summary = last_assistant.splitlines()[0][:120]
            routing_query = f"Previous answer was: {last_assistant_summary}\n\nCurrent question: {current_q}"
        else:
            routing_query = current_q

        logger.debug(f"  Routing question: {routing_query}")

        result = router_chain.invoke({"query": routing_query})
        args = result.tool_calls[0]["args"]

        routes          = args["routes"]
        answer_strategy = args["answer_strategy"]
        retrieval_query = args["retrieval_query"]
        reasoning       = args["reasoning"]

        # Safety net: if a project ID appears anywhere in the full conversation
        # history, make sure proj_vectorstore is included for follow-up questions.
        if extract_project_id_from_history(full_question) and "proj_vectorstore" not in routes:
            routes.append("proj_vectorstore")
            logger.info("  Safety net: added proj_vectorstore from conversation history")

        logger.info(f"  answer_strategy : {answer_strategy}")
        logger.info(f"  routes          : {routes}")
        logger.info(f"  retrieval_query : {retrieval_query}")
        logger.info(f"  reasoning       : {reasoning}")

        return {
            "routes":          routes,
            "answer_strategy": answer_strategy,
            "retrieval_query": retrieval_query,
            "reasoning":       reasoning,
        }

    # ------------------------------------------------------------------
    # Pandas aggregate node — unchanged from original
    # ------------------------------------------------------------------

    def pandas_query_node(state: dict) -> dict:
        logger.info("NODE: Pandas Aggregate Query")
        question = state["question"]

        def get_score(x):
            if isinstance(x, dict):
                return x.get("score")
            try:
                v = float(x)
                return int(v) if v == int(v) else v
            except (TypeError, ValueError):
                return None

        schema_info = f"""
        You have a pandas DataFrame called `proj_df` with {len(proj_df)} rows.

        ACTUAL columns: {list(proj_df.columns)}

        SCORE columns (plain integers — ONLY these should ever have int() applied):
        {[c for c in proj_df.columns if c.endswith('_score')]}

        DESCRIPTION columns (plain text strings — NEVER apply int() to these):
        {[c for c in proj_df.columns if c.endswith('_desc')]}

        IMPORTANT SCORING COLUMN DISAMBIGUATION:
        - Every scoring category has TWO columns: a region_ version and a lead_proponent_ version
        - region_readiness_support_score         → the REGION score (default)
        - lead_proponent_readiness_support_score → the LEAD PROPONENT score
        - Same pattern applies to all scoring pairs:
            region_mission_alignment_score       vs lead_proponent_mission_alignment_score
            region_operational_cost_score        vs lead_proponent_operational_cost_score
            region_severity_statement_score      vs lead_proponent_severity_statement_score
            region_urgency_statement_score       vs lead_proponent_urgency_statement_score
        - If the user does not specify "lead proponent", ALWAYS use the region_ version

        KEY COLUMN NOTES:
        - 'installation': ends in country code (' IT'=Italy, ' GR'=Greece, ' SP'=Spain, ' JA'=Japan, ' BH'=Bahrain)
        - 'CWE': cost in THOUSANDS of dollars (100000 = $100M)
          IMPORTANT: column name is uppercase 'CWE' — never write 'cwe' or 'Cwe'
        - SCORING COLUMNS ARE PLAIN INTEGERS — filter directly: proj_df['region_mission_alignment_score'] >= 3
        - A value of -1 means score not provided — exclude with col > 0
        - Do NOT define any functions — use direct pandas operations or lambdas only
        - IMPORTANT: str.endswith() does NOT accept a list — use a tuple instead:
          CORRECT:   proj_df['installation'].str.endswith((' IT', ' GR'))
          INCORRECT: proj_df['installation'].str.endswith([' IT', ' GR'])
        - For multiple countries, always use a tuple in str.endswith()
        - 'project_id': string like 'P309', 'RM23-0509'
        - ALL column names are case-sensitive. Always use the exact names from the 
          ACTUAL columns list above. Never lowercase a column name.
        """

        code_prompt = (
            schema_info
            + '\n\nWrite Python/pandas code to answer this question: "'
            + question
            + '''"\n\nRULES:
                - SCORE COLUMN DEFAULT: when the user mentions a score category (e.g. "readiness score",
                "mission alignment score", "operational cost score") without specifying "lead proponent",
                ALWAYS use the region_ column (e.g. region_readiness_support_score).
                Only use lead_proponent_ columns if the user explicitly says "lead proponent score".
                This rule applies to counts, filters, lists, and any other query involving scores.
                NEVER switch column between a count query and its follow-up list query.
                - For SIMPLE results (single number, count, or short sentence):
                set result = "some string"
                Example: result = "There are 5 projects in Italy."
                - For TABLE results (any list of projects or multi-column data):
                set result_df as a pandas DataFrame — do NOT build a markdown string
                Example:
                    filtered = proj_df[mask]
                    result_df = filtered[['project_id', 'title', 'installation', 'CWE']].copy()
                    result_df = result_df.rename(columns={
                        'project_id': 'Project ID',
                        'title': 'Title',
                        'installation': 'Installation',
                        'CWE': 'CWE ($K)'
                    })
                    result = None  # always set result = None when using result_df

                - COLUMN FORMATTING (apply these transformations to result_df before returning):
                    * CWE column: format each value as a dollar amount in thousands with comma separator
                    e.g. 7147 becomes "$7,147K", null becomes "N/A"
                    Use: result_df['CWE ($K)'] = result_df['CWE ($K)'].apply(
                        lambda x: ("$" + f"{int(x):,}" + "K") if pd.notna(x) else "N/A")
                    * Score columns end in '_score' (they are integers, -1 = not provided).
                    ONLY apply int() formatting to columns whose name ends in '_score'.
                    NEVER apply int() to columns ending in '_desc' — those are plain text strings
                    and int() will crash on them.
                    Use: result_df['col'] = result_df['col'].apply(
                        lambda x: "N/P" if pd.isna(x) or x == -1 else str(int(x)))

                    * SCORE COLUMN INCLUSION RULE: if you apply formatting to a score column on result_df,
                    that column MUST be included in the result_df column selection list BEFORE formatting.
                    WRONG:
                        result_df = filtered[['project_id', 'title', 'installation', 'CWE']].copy()
                        result_df['region_mission_alignment_score'] = result_df['region_mission_alignment_score'].apply(...)
                    CORRECT:
                        result_df = filtered[['project_id', 'title', 'installation', 'CWE',
                                            'region_mission_alignment_score']].copy()
                        result_df['region_mission_alignment_score'] = result_df['region_mission_alignment_score'].apply(...)
                    Then rename in the rename() call:
                        result_df = result_df.rename(columns={
                            'project_id': 'Project ID',
                            'title': 'Title',
                            'installation': 'Installation',
                            'CWE': 'CWE ($K)',
                            'region_mission_alignment_score': 'Mission Align Score'
                        })
                    * ALL column names are case-sensitive — always use exact names from the
                    ACTUAL columns list above. Never lowercase a column name (e.g. 'CWE' not 'cwe')

                - FOLLOW-UP COLUMN RULE:
                    * If the question adds a column to a previous list, rebuild the full
                    result_df from scratch using the same filter — never reference a
                    previous result variable

                - Do NOT import anything — pandas is available as pd
                - Do NOT define any functions — use direct pandas operations or lambdas only
                - Output ONLY the code, no explanation, no markdown fences
                '''
                )

        response = llm_gen_tools.invoke([{"role": "user", "content": code_prompt}])
        code = response.content.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        logger.debug(f"  Generated pandas code:\n{code}")

        local_vars = {"proj_df": proj_df.copy(), "pd": pd, "json": json, "get_score": get_score}
        try:
            exec(code, local_vars)

            result_df = local_vars.get("result_df", None)
            result_str = local_vars.get("result", None)

            if result_df is not None and isinstance(result_df, pd.DataFrame):
                # Serialize the dataframe for the UI to render as st.dataframe
                result_str = "DATAFRAME:" + result_df.to_json(orient="records")
                logger.info(f"  DataFrame result: {len(result_df)} rows x {len(result_df.columns)} cols")
            elif result_str is not None:
                result_str = str(result_str)
            else:
                result_str = "Could not compute result."

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
    
    def pandas_list_node(state: dict) -> dict:
        logger.info("NODE: Pandas List Query")
        question = state["question"]

        # Extract current question only
        if "Current question:" in question:
            current_q = question.split("Current question:")[-1].strip()
        else:
            current_q = question

        schema_info = f"""
        You have access to a pandas DataFrame called proj_df with {len(proj_df)} rows.

        ACTUAL columns: {list(proj_df.columns)}

        KEY COLUMN NOTES:
        - 'installation': ends in country code (' IT'=Italy, ' GR'=Greece, ' SP'=Spain)
        - 'CWE': cost in THOUSANDS of dollars
        - Score columns end in '_score' (integers, -1 = not provided, use region_ by default)
        - Description columns end in '_desc' (plain text, never apply int() to these)
        - ALL column names are case-sensitive
        """

        code_prompt = (
            schema_info
            + '\n\nWrite Python/pandas code to answer this question: "'
            + current_q
            + '''"\n\nRULES:
                - Filter proj_df to the relevant projects using pandas
                - Set result as a plain text BULLETED LIST string (use "• " prefix for each item)
                - Each bullet must include: Project ID, Title, and a 1-2 sentence explanation
                of why that project meets the criteria in the question
                - Base explanations on actual data from the row fields (scope, score columns,
                description columns, installation, CWE, etc.) — do not invent information
                - Do NOT build a DataFrame or markdown table — result must be a plain string
                - Do NOT import anything — pandas is available as pd
                - Do NOT define any functions — use direct pandas operations only
                - Output ONLY the code, no explanation, no markdown fences

                EXAMPLE FORMAT:
                    filtered = proj_df[some_mask]
                    result = ""
                    for _, row in filtered.iterrows():
                        pid   = row["project_id"]
                        title = row.get("title", "N/A")
                        score = row.get("region_mission_alignment_score", -1)
                        desc  = row.get("region_mission_alignment_desc", "")
                        result += f"• {pid} — {title}: Region mission alignment score is {score}. {desc[:1000]}...\\n\\n"
                    if not result:
                        result = "No projects found matching that criteria."
                '''
                        )

        response = llm_gen_tools.invoke([{"role": "user", "content": code_prompt}])
        code = response.content.strip().removeprefix("```python").removeprefix("```").removesuffix("```").strip()
        logger.debug(f"  Generated pandas list code:\n{code}")

        local_vars = {"proj_df": proj_df.copy(), "pd": pd, "json": json}
        try:
            exec(code, local_vars)
            result_str = str(local_vars.get("result", "Could not compute result."))
        except Exception as e:
            result_str = f"Error running that query: `{e}`"
            logger.error(f"  Pandas list exec error: {e}\nCode:\n{code}")

        logger.info(f"  Pandas list result preview: {result_str[:200]}")
        return {"generation": result_str}

    # ------------------------------------------------------------------
    # Project ID helpers — kept as-is, still useful for safety nets
    # ------------------------------------------------------------------

    def extract_project_id(question: str) -> str | None:
        match = re.search(
            r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4}|B\d{3,4})\b',
            question, re.IGNORECASE
        )
        return match.group(0).upper() if match else None

    def extract_project_id_from_history(question: str) -> str | None:
        all_matches = re.findall(
            r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4}|B\d{3,4})\b',
            question, re.IGNORECASE
        )
        return all_matches[-1].upper() if all_matches else None

    # ------------------------------------------------------------------
    # Hybrid retrieval helpers — unchanged
    # ------------------------------------------------------------------

    def hybrid_retrieve(store, query, k=6, filter_dict=None):
        dense_docs = store.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_dict
        )
        collection = store._collection.get(include=["documents", "metadatas"])
        bm25_docs = [
            Document(page_content=doc, metadata=meta)
            for doc, meta in zip(collection["documents"], collection["metadatas"])
        ]
        sparse_retriever = BM25Retriever.from_documents(documents=bm25_docs)
        sparse_docs = sparse_retriever.invoke(query)

        def normalize(arr):
            arr = np.array(arr)
            return (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)

        sparse_scores = normalize([1.0] * len(sparse_docs))

        fused = []
        for (doc, d_score), s_score in zip(dense_docs, sparse_scores):
            fused_score = 0.6 * d_score + 0.4 * s_score
            fused.append((doc, float(fused_score)))

        for doc in sparse_docs[len(dense_docs):]:
            fused.append((doc, 0.4))

        fused_sorted = sorted(fused, key=lambda x: x[1], reverse=True)
        return fused_sorted[:k]

    def cross_encoder_rerank(query: str, docs_with_scores: list, top_k: int = 6):
        pairs = [(query, doc.page_content) for doc, _ in docs_with_scores]
        scores = cross_encoder.predict(pairs)
        reranked = list(zip([doc for doc, _ in docs_with_scores], scores))
        reranked_sorted = sorted(reranked, key=lambda x: x[1], reverse=True)
        return [(doc, float(score)) for doc, score in reranked_sorted[:top_k]]

    # ------------------------------------------------------------------
    # Semantic retrieval node
    # Now uses state["retrieval_query"] (set by the router) instead of
    # rebuilding the query from keyword lists.
    # ------------------------------------------------------------------

    def semantic_retrieve_w_scores(state: dict) -> dict:
        logger.info("NODE: Semantic Retrieve")
        routes          = state.get("routes", [])
        full_question   = state["question"]
        retrieval_query = state.get("retrieval_query", full_question)  # ← set by router
        k               = state.get("k", 6)

        logger.debug(f"  retrieval_query: {retrieval_query}")

        # Resolve project ID for vectorstore filter.
        # For semantic_rag: check the current question first, then fall back to
        # conversation history (handles pronoun references like "it"/"this project").
        # For pandas_aggregate: only look at the current question — scanning history
        # would latch onto project IDs that appear in a previous table result and
        # incorrectly filter the vectorstore to a single project.
        answer_strategy = state.get("answer_strategy", "semantic_rag")

        if answer_strategy == "semantic_rag":
            project_id = (
                extract_project_id(retrieval_query)
                or extract_project_id_from_history(full_question)
            )
        else:
            project_id = extract_project_id(retrieval_query)

        if project_id:
            logger.info(f"  project_id resolved: {project_id} (strategy={answer_strategy})")

        # Ensure proj_vectorstore is in routes if a project ID was found
        if project_id and "proj_vectorstore" not in routes:
            routes = list(routes) + ["proj_vectorstore"]
            logger.info(f"  Safety net: added proj_vectorstore for {project_id}")

        filter_dict = {"project_id": {"$eq": project_id}} if project_id else None
        logger.info(f"  filter: {filter_dict}")

        store_map = {
            "proj_vectorstore":    proj_vectorstore,
            "strat26_vectorstore": strat26_vectorstore,
            "strat28_vectorstore": strat28_vectorstore,
        }

        stores = [store_map[r] for r in routes if r in store_map]
        logger.info(f"  Stores to query: {[r for r in routes if r in store_map]}")

        if not stores:
            return {"documents": []}

        all_docs = []
        for store in stores:
            filt = filter_dict if store is proj_vectorstore else None
            docs = hybrid_retrieve(store=store, query=retrieval_query, k=k * 3, filter_dict=filt)
            logger.info(f"  Hybrid retrieved {len(docs)} docs from store")
            reranked_docs = cross_encoder_rerank(
                query=retrieval_query, docs_with_scores=docs, top_k=k
            )
            all_docs.extend(reranked_docs)

        return {"documents": all_docs}

    # ------------------------------------------------------------------
    # RAG generation prompt — unchanged
    # ------------------------------------------------------------------

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
    - If the user asks how a project WOULD align, COULD score, or asks to RE-SCORE under 
      POM28 guidance, you are being asked to GENERATE a new estimated score and justification.
    - POM28 guidance contains ONLY updated NSS/NDS strategic themes — there are no new CNIC 
      scoring criteria for POM28. The CNIC scoring rubric (1-5 scale definitions) from POM26 
      remains in effect and should always be used.
    - The correct approach for POM28 re-scoring is:
        1. Use the POM26 CNIC scoring rubric (0-4) as the scoring framework — this does not change
        2. Evaluate the project against the UPDATED NSS/NDS themes from POM28 documents
        3. A project scores higher if it aligns strongly with POM28 NSS/NDS priorities 
           (e.g. peace through strength, homeland defense, warfighter lethality, forward 
           posture, alliance resilience) even if those themes differ from POM26
    - Use the project's scope, mission, impact, and existing POM26 justifications as the 
      factual basis — the score changes only if POM28 NSS/NDS themes would cause a 
      re-evaluation of that alignment.
    - Clearly label your output as an ESTIMATED score under POM28, not the official recorded score.
    - Format as: "**Estimated POM28 Score: [0-4]** — [justification referencing the project 
      details and the specific POM28 NSS/NDS themes that support or change the score]"
    - Only refuse to score if NO strategy documents whatsoever appear in the retrieved context.
    - Any retrieved NSS/NDS text is sufficient to generate an estimate — do not require 
      CNIC scoring rubric text to be present before proceeding.

ANSWERING RULES - GENERATING NEW JUSTIFICATION STATEMENTS:
- When asked to generate a new justification block (e.g. region mission alignment statement,
  impact if not provided, scope, severity statement, urgency statement, readiness support
  statement, operational cost statement), match the length and style of existing statements
  for that same field type in the retrieved documents.
- Before writing, scan the retrieved documents for examples of that field type and use them
  to calibrate your response length and tone. For example:
    * region_mission_alignment_desc entries are typically 3-5 sentences covering strategic
      gaps addressed, COCOMs supported, and capability contributions
    * impact_if_not_provided entries are typically 2-4 sentences describing mission
      degradation, workarounds required, and consequences of deferral
    * scope entries are typically 2-3 sentences describing what is being constructed
      or repaired and the key systems involved
    * region_severity_statement_desc entries are typically 2-3 sentences describing
      current facility condition and mission impact
    * region_urgency_statement_desc entries are typically 2-3 sentences explaining
      why investment is needed now and consequences of delay
    * region_readiness_support_desc entries are typically 3-4 sentences describing
      the BFR gap, current workarounds, and how the project closes the gap
    * region_operational_cost_desc entries are typically 2-3 sentences with a
      specific cost figure, payback period, and cost savings rationale
- Do NOT write a single sentence when the field type typically spans a paragraph.
- Do NOT write multiple paragraphs when the field type is typically 2-3 sentences.
- Write in the same terse, direct, third-person style used in official Navy PDS documents —
  avoid conversational language, hedging, or AI-sounding phrases like "it is worth noting".

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

    # ------------------------------------------------------------------
    # Generate response node — unchanged
    # ------------------------------------------------------------------

    def generate_response(state: dict) -> dict:
        logger.info("NODE: Generate Response")
        question  = state["question"]
        documents = state["documents"]

        logger.debug(f"  Question: {question}")
        logger.debug(f"  Using {len(documents)} documents for context")

        context_parts = []
        for doc, score in documents:
            project_id   = doc.metadata.get("project_id", "")
            installation = doc.metadata.get("installation", "")
            header = f"[project_id: {project_id}, installation: {installation}]\n" if project_id else ""
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
                "contain absolutely no relevant information. Simply provide the answer without any "
                "additional commentary or explanation. Be direct and to the point."
            )

        logger.debug(f"  Context length: {len(context)} chars")
        logger.debug(f"  Full context:\n{context}")

        prompt = rag_template.invoke({
            "query": question,
            "context": context,
            "answer_instructions": answer_instructions
        })

        logger.info(f"  Prompt created with {len(prompt.messages)} messages")
        for m in prompt.messages:
            logger.debug(f"  PROMPT MSG [{m.type}]:\n{m.content}")

        logger.info("  Invoking LLM...")
        msg = llm_gen_tools.invoke(prompt.messages)

        generation_text = getattr(msg, "content", "") or ""
        logger.info(f"  Generated response ({len(generation_text)} chars)")
        logger.debug(f"  Response preview: {generation_text[:200]}...")

        return {"generation": generation_text, "gen_attempts": 1}

    generate_response_runnable = RunnableLambda(generate_response, name="generate_response")

    # ------------------------------------------------------------------
    # Check answer node
    # Removed the GENERATIVE_KEYWORDS list — the grader skip is now
    # driven by answer_strategy instead.
    # ------------------------------------------------------------------

    idk_phrases = ["I don't know", "I do not know"]

    def check_answer_node(state: dict) -> dict:
        logger.info("NODE: Check Answer")

        question        = state.get("question", "")
        generation      = state.get("generation", "")
        documents       = state.get("documents", [])
        attempts        = state.get("gen_attempts", 0)
        max_retries     = state.get("max_retries", 3)
        answer_strategy = state.get("answer_strategy", "semantic_rag")

        # IDK check
        if any(phrase.lower() in generation.lower() for phrase in idk_phrases):
            logger.info("  IDK detected")
            if attempts < max_retries:
                return {"route_after_check": "retry"}
            return {"route_after_check": "final"}

        # Multiple choice enforcement
        if bool(re.search(r'\b[A-D]\)', question)):
            if not re.search(r'\b[A-D]\) ?\S+', generation):
                logger.info("  No valid answer choice found")
                if attempts < max_retries:
                    return {"route_after_check": "retry"}
            logger.info("  Multiple choice answer found -> final")
            return {"route_after_check": "final"}

        # Skip grader for generative/alignment questions.
        # Previously this was a keyword list — now we use the router's answer_strategy.
        # "semantic_rag" answers involve synthesis across documents; the grader will
        # incorrectly reject valid answers because generated text won't match
        # retrieved chunks verbatim. Only run the grader for simple factual RAG answers
        # where we can verify grounding. For now, treat all semantic_rag as generative.
        #
        # If you want finer control, the router could return a third strategy value
        # like "factual_rag" for simple lookups vs "generative_rag" for alignment questions.
        if answer_strategy == "semantic_rag":
            logger.info("  semantic_rag answer — skipping grader, returning final")
            return {"route_after_check": "final"}

        # LLM grounding check for factual questions (currently only pandas_aggregate
        # would reach here, but pandas_aggregate routes to END directly so this
        # is a safety net for any future strategy types)
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

    # -----------------------------------------------------------------------
    # GRAPH EDGES
    # route_question_edge is now a one-liner — no keyword detection needed.
    # -----------------------------------------------------------------------

    def route_question_edge(state: dict) -> str:
        strategy = state.get("answer_strategy", "semantic_rag")
        logger.info(f"EDGE: route_question_edge -> {strategy}")
        return strategy

    def decide_after_check(state: dict) -> str:
        return state.get("route_after_check", "final")

    # -----------------------------------------------------------------------
    # BUILD WORKFLOW
    # -----------------------------------------------------------------------

    workflow = StateGraph(GraphState)

    workflow.add_node("init_state",                    init_graph_state)
    workflow.add_node("route_question",                route_question_node)
    workflow.add_node("pandas_aggregate",              pandas_query_node)
    workflow.add_node("pandas_list",                   pandas_list_node)
    workflow.add_node("semantic_retriever_and_grader", semantic_retrieve_w_scores)
    workflow.add_node("generate_response",             generate_response)
    workflow.add_node("check_answer",                  check_answer_node)

    workflow.set_entry_point("init_state")
    workflow.add_edge("init_state", "route_question")

    workflow.add_conditional_edges(
        "route_question",
        route_question_edge,
        {
            "pandas_aggregate": "pandas_aggregate",
            "pandas_list":      "pandas_list",
            "semantic_rag":     "semantic_retriever_and_grader",
        }
    )

    workflow.add_edge("pandas_aggregate", END)
    workflow.add_edge("pandas_list",      END)
    workflow.add_edge("semantic_retriever_and_grader", "generate_response")
    workflow.add_edge("generate_response",             "check_answer")

    workflow.add_conditional_edges(
        "check_answer",
        decide_after_check,
        {
            "final": END,
            "retry": "semantic_retriever_and_grader",
        }
    )

    app = workflow.compile()

    # -----------------------------------------------------------------------
    # INVOKE
    # -----------------------------------------------------------------------

    MAX_HISTORY_MSG_CHARS = 1000

    if history:
        history_lines = []
        for m in history[-4:]:
            content = m["content"]
            # For dataframe messages, convert to a brief summary for routing context
            if m.get("is_dataframe") and content.startswith("DATAFRAME:"):
                try:
                    records = json.loads(content[len("DATAFRAME:"):])
                    df_hist = pd.DataFrame(records)
                    # Give the router just the shape and first row as context
                    content = f"[Table: {len(df_hist)} rows x {len(df_hist.columns)} cols, columns: {list(df_hist.columns)}]"
                except Exception:
                    content = "[Table result]"
            elif len(content) > MAX_HISTORY_MSG_CHARS:
                content = content[:MAX_HISTORY_MSG_CHARS] + "..."
            history_lines.append(f"{m['role'].upper()}: {content}")
        history_text = "\n".join(history_lines)
        enriched_question = f"Conversation so far:\n{history_text}\n\nCurrent question: {user_msg}"
    else:
        enriched_question = user_msg

    result = app.invoke({"question": enriched_question})

    generation = result.get("generation", "I don't know.")
    generation = re.sub(r'<think>.*?</think>', '', generation, flags=re.DOTALL).strip()
    documents  = result.get("documents", [])

    return generation, documents


# --------------------------------------
# CHAT ASSISTANT UI  (unchanged)
# --------------------------------------

st.markdown("---")
st.subheader("Ask the dashboard assistant...")

if not is_llm_configured():
    st.info(
        "LLM assistant not configured. "
        "If you're running this app yourself, add your NVIDIA API key to "
        "`.streamlit/secrets.toml` (see README & Tutorial_Dashboard.md for instructions)."
    )
else:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_dataframe") and msg["content"].startswith("DATAFRAME:"):
                try:
                    records = json.loads(msg["content"][len("DATAFRAME:"):])
                    st.dataframe(pd.DataFrame(records), use_container_width=True)
                except Exception:
                    st.markdown("*(table unavailable)*")
            else:
                st.markdown(msg["content"])

    user_msg = st.chat_input(
        "Ask me a question about a MILCON or FSRM project (only NAVEUR/NAVAF for now!)"
    )

    if user_msg:
        st.session_state.messages.append({"role": "user", "content": user_msg})

        with st.chat_message("user"):
            st.markdown(user_msg)

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

                # Check if the reply is a serialized dataframe
                if isinstance(assistant_reply, str) and assistant_reply.startswith("DATAFRAME:"):
                    try:
                        records = json.loads(assistant_reply[len("DATAFRAME:"):])
                        df_display = pd.DataFrame(records)
                        st.dataframe(df_display, use_container_width=True)
                    except Exception as e:
                        st.markdown(f"Error rendering table: `{e}`")
                else:
                    # Ensure bullet points each render on their own line
                    st.markdown(assistant_reply.replace("\n•", "\n\n•"))

                # Show sources in collapsible expander
                if source_docs:
                    with st.expander("📄 Sources", expanded=False):
                        seen = set()
                        for doc, score in source_docs:
                            pid         = doc.metadata.get("project_id")
                            title       = doc.metadata.get("title", "")
                            install     = doc.metadata.get("installation", "")
                            source_file = doc.metadata.get("source", "")
                            filename    = source_file.split("/")[-1] if source_file else "Unknown"

                            dedup_key = pid if pid else filename
                            if dedup_key in seen:
                                continue
                            seen.add(dedup_key)

                            if pid:
                                st.markdown(f"**{pid}** — {title}  \n`{install}` | File: `{filename}`")
                            else:
                                st.markdown(f"📘 **{filename}**")

        st.session_state.messages.append(
            {"role": "assistant", "content": assistant_reply, "is_dataframe": isinstance(assistant_reply, str) and assistant_reply.startswith("DATAFRAME:")}
        )
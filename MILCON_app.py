# Luke Holmes

# Intro page to Vietnam Bombing Dashboard
# This is what's known as the "Entry Point" page for the Streamlit app. It provides an 
# overview of the dashboard, its purpose, and the data source. It also sets up the page
# configuration and title. The scripts in the "pages" folder contain the subsequent pages
# of the dashboard, which are accessible via the sidebar navigation in Streamlit.

# I used ChatGPT-5.1, Copilot, and Streamlit's documentation to help write this code, so I'll
# narrate this code to explain its functionality and purpose.


import polars as pl
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

st.markdown(
    """
    ### Dashboard Overview
    """,
    unsafe_allow_html=True              # Render HTML safely - not plain text
)

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

# Initialize vectorstores (adjust paths/collection names to match your setup)
hfe = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load vectorstores:
proj_vectorstore_path: str = "./databases/proj"
proj_vectorstore = load_vectorstore(proj_vectorstore_path, embedding_function=baseline_hfe)

strat26_vectorstore_path: str = "./databases/strat26"
strat26_vectorstore = load_vectorstore(strat26_vectorstore_path, embedding_function=baseline_hfe)

strat28_vectorstore_path: str = "./databases/strat28"
strat28_vectorstore = load_vectorstore(strat28_vectorstore_path, embedding_function=baseline_hfe)


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

        question = state["question"]
        logger.debug(f"  Question to route: {question}")

        result = router_chain.invoke({"query": question})
        routes = result.tool_calls[0]["args"]["routes"]

        logger.info(f"  Routing decision: {routes}")
        return {"routes": routes}



    # Build the retrieval node
    def extract_project_id(question: str) -> str | None:
        match = re.search(r'\b(P\d{3,4}|RM\d{2}-\d{4}|NF\d{2}-\d{4}|ST\d{2}-\d{4})\b', question, re.IGNORECASE)
        return match.group(0).upper() if match else None  # normalize to uppercase

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
    - Documents are JSON-formatted and contain fields like installation (the project's location/base), title, region, scope, CWE (cost), COCOM, and scoring fields — use these fields to answer questions.
    - Project IDs like "P314", "RM16-0799", "NF22-1234" refer to the "project_id" field in the documents. When a user asks about "P314", they mean the document where project_id = "P314". Similarly, if asked p314 or rm16-0799, it still refers to the same project_id field in the documents - just normalized to uppercase.
    - The "installation" field is the physical location of the project. If asked "where is" a project, answer using the installation name and infer the city/country if it is recognizable (e.g., "NAVSUPPACT NAPLES IT" = Naples, Italy).
    - Scoring fields come in pairs: "region_<category>" and "lead_proponent_<category>" (e.g., region_mission_alignment and lead_proponent_mission_alignment). Each has a "score" and "description" subfield. A null value means no score or statement was provided.

    MISSING‑EVIDENCE RULES
    - If no documents are retrieved and the question concerns Navy facilities projects, respond with "I don't know. No relevant project documents were retrieved to answer this question. Are you sure you have the correct project ID?"
    - If no documents are retrieved and the question is general (not about Navy facilities), you may answer using general knowledge.

    ANSWERING RULES
    - Answer in 1-3 sentences. Be direct and concise.
    - SCORING RULE: If the user asks for a score in a category without specifying "region" or "lead proponent", always report BOTH the region score and the lead proponent score. If either is null, state "not provided". Example: "The region mission alignment score is 4. No lead proponent score was provided."
    - SCORING RULE: If the user specifies "region" or "lead proponent" explicitly, return only that one score.
    - Never say "I don't know" if the documents contain any relevant information — extract and use what is there. If a field is null, say "not provided" rather than "I don't know".
    - Cite the specific field or value you are drawing from (e.g., "According to the region_severity_statement field...").
    - When multiple document domains are retrieved, integrate them into a single coherent answer.
    - if askedabout a cost, or reporting CWE information, automatically convert to thousands of dollars (e.g. "CWE ($000) is $68140, or $68,140,000" rather than "68140"). If the CWE field is null, say "CWE information was not provided for this project."
    - When POM26 and POM28 guidance conflict, clearly distinguish between them.
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

    generation = result.get("generation", "I don't know.")
    # Strip reasoning model <think> blocks
    generation = re.sub(r'<think>.*?</think>', '', generation, flags=re.DOTALL).strip()
    return generation

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
            with st.spinner("Thinking..."):                     # Show a "thinking" spinner while waiting for LLM response
                try:
                    assistant_reply = run_llm_intro(            # Call our LLM helper function
                        user_msg=user_msg,                      # User's message
                        history=st.session_state.messages[:-1]  # Prior chat history - exclude current user message...
                    )                                           # so it doesn't appear twice
                
                except Exception as e:                          # If there's an error calling the LLM API
                    assistant_reply = f"Sorry, I couldn't reach the LLM API: `{e}`"

                st.markdown(assistant_reply)                    # Display the LLM's reply in the chat

        # Save assistant reply to history
        st.session_state.messages.append(
            {"role": "assistant", 
             "content": assistant_reply}
        )
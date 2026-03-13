# RAG components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LangChain tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# LLM
from langchain_openai import ChatOpenAI

# Random Question Sampling
import random
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------
# DO NOT MODIFY ANYTHING ABOVE eval_rag_chain_proj_query
# ----------------------------------------------------

# Generative LLM chat client
llm = ChatOpenAI(
    model="TRAC-MTRY/traclm-v4-7b-instruct",
    openai_api_key="sk-UtrV9i5fFenmG6hvMss71A",
    openai_api_base="http://trac-malenia.ern.nps.edu:8080/inference/v1",
    temperature=0
)

# Basic prompt template
prompt_template = ChatPromptTemplate([
    (
        "system",
        "You are a helpful asssistant that uses provided context to answer user questions: {context}\n",
    ),
    (
        "human",
        "{prompt}"
    )
])

# Embedding model
embed_model_name = "all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}

baseline_hfe = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

# Access baseline vector store
baseline_vectorstore = Chroma(
    embedding_function=baseline_hfe,
    persist_directory="./databases/chroma_baseline",
)

# Make retriever a runnable so it can be chained
baseline_retriever = RunnableLambda(
    lambda a_query: baseline_vectorstore.similarity_search_with_relevance_scores(
        a_query, k=1
    )
)

# Set up chain to get output
parser = StrOutputParser()
generation_chain = prompt_template | llm | parser

# Set up RAG chain
baseline_rag_chain = (
    {
        "context": baseline_retriever,
        "prompt": RunnableLambda(lambda x: x),
    }
    | generation_chain
)


# ----------------------------------------------------
# QUESTION BANK
# ----------------------------------------------------

q_a_pairs = [
    # RM21-0367
    {
        "question": "What is the current working estimate (CWE) for project RM21-0367, in $K?",
        "options": ["A) 5,828", "B) 3,188", "C) 4,684", "D) 11,470", "E) 28,001"],
        "answer": "B) 3,188"
    },
    {
        "question": "What is the Overall Capacity Rating for RM21-0367?",
        "options": ["A) 80", "B) 100", "C) 45", "D) 88", "E) 65"],
        "answer": "C) 45"
    },
    # RM20-0438
    {
        "question": "What is the condition rating of the facility associated with project RM20-0438?",
        "options": ["A) 73", "B) 64", "C) 89", "D) 76", "E) 83"],
        "answer": "C) 89"
    },
    {
        "question": "What score did RM20-0438 receive for Lead Proponent Readiness Support?",
        "options": ["A) 1", "B) 2", "C) 3", "D) 4", "E) 0"],
        "answer": "B) 2"
    },
    # RM18-1324
    {
        "question": "How many outages per year due to urgent or emergency repairs were reported for RM18-1324?",
        "options": ["A) 5", "B) 10", "C) 15", "D) 20", "E) 25"],
        "answer": "B) 10"
    },
    {
        "question": "What is the COCOM for project REPAIR POTABLE WATER NETWORK SYSTEM AT NSA-I?",
        "options": ["A) EUCOM", "B) INDOPACOM", "C) CENTCOM", "D) AFRICOM", "E) NORTHCOM"],
        "answer": "C) CENTCOM"
    },
    # NF20-0826
    {
        "question": "What is the RAC rating for project NF20-0826 PORT OPERATIONS WAREHOUSE?",
        "options": ["A) I", "B) II", "C) III", "D) IV", "E) Not listed"],
        "answer": "B) II"
    },
    {
        "question": "What is the Project CCN for NF20-0826?",
        "options": ["A) 72111", "B) 84210", "C) 73025", "D) 44110", "E) 14125"],
        "answer": "D) 44110"
    },
    # ST18-1369
    {
        "question": "What is the average PCI rating of NS Rota Airfield as reported in project ST18-1369?",
        "options": ["A) 55", "B) 67", "C) 42", "D) 35", "E) 71"],
        "answer": "C) 42"
    },
    {
        "question": "What score did ST18-1369 receive for Region Operational Cost?",
        "options": ["A) 0", "B) 1", "C) 2", "D) 3", "E) 4"],
        "answer": "B) 1"
    },
    # RM16-0799
    {
        "question": "What is the estimated annual sunk cost due to inefficiencies at the AIMD facility associated with RM16-0799?",
        "options": ["A) $500K", "B) $1M", "C) $2M", "D) $5M", "E) $10M"],
        "answer": "C) $2M"
    },
    {
        "question": "What is the Overall Capacity Rating for RM16-0799?",
        "options": ["A) 75", "B) 50", "C) 88", "D) 26", "E) 40"],
        "answer": "D) 26"
    },
    # RM15-0946
    {
        "question": "How many emergency/urgent trouble calls were placed for the facility in project RM15-0946?",
        "options": ["A) 42", "B) 60", "C) 87", "D) 100", "E) 23"],
        "answer": "C) 87"
    },
    {
        "question": "How many two-person rooms does Building 266 provide per the RM15-0946 project sheet?",
        "options": ["A) 100", "B) 172", "C) 219", "D) 344", "E) 266"],
        "answer": "B) 172"
    },
    # P314
    {
        "question": "What is the CWE (in $K) for project P314?",
        "options": ["A) 85,920", "B) 41,820", "C) 68,140", "D) 231,370", "E) 28,170"],
        "answer": "D) 231,370"
    },
    {
        "question": "What is the current capacity percentage of existing spaces relative to the requirement for P314?",
        "options": ["A) 40%", "B) 50%", "C) 61%", "D) 75%", "E) 85%"],
        "answer": "C) 61%"
    },
    # P738
    {
        "question": "How old is the existing fire station facility referenced in project P738?",
        "options": ["A) 25 years", "B) 35 years", "C) 47 years", "D) 60 years", "E) 30 years"],
        "answer": "C) 47 years"
    },
    {
        "question": "What is the score for Region Severity Statement for project P738?",
        "options": ["A) 4", "B) 1", "C) 2", "D) 3", "E) 0"],
        "answer": "D) 3"
    },
    # P222
    {
        "question": "How many 2+0 modules will the new facility constructed under P222 contain?",
        "options": ["A) 172", "B) 199", "C) 219", "D) 333", "E) 428"],
        "answer": "B) 199"
    },
    {
        "question": "What is the Tenant Capacity Rating for P222?",
        "options": ["A) 85", "B) 100", "C) 50", "D) 23", "E) 71"],
        "answer": "D) 23"
    },
    # P1121
    {
        "question": "Where is project P1121 EDI: Prepositioned Logistics Facility located?",
        "options": ["A) NAVSTA Rota, Spain", "B) NAVSUPPACT Bahrain", "C) NSA Naples, Italy", "D) NAS Sigonella, Italy", "E) VARLOCS"],
        "answer": "E) VARLOCS"
    },
    {
        "question": "What is the lead proponent for project P1121?",
        "options": ["A) CNIC N9", "B) NAVFAC", "C) NECE", "D) NAE", "E) CNIC N3"],
        "answer": "C) NECE"
    },
    # P1413
    {
        "question": "What is the estimated annual energy cost savings for project P1413 Energy Resilience and Cyber Security of Critical Loads?",
        "options": ["A) $128K", "B) $592.59K", "C) $728K", "D) $1.1M", "E) $7.18M"],
        "answer": "B) $592.59K"
    },
    {
        "question": "What Region Severity score did P1413 receive?",
        "options": ["A) 1", "B) 2", "C) 3", "D) 4", "E) 0"],
        "answer": "B) 2"
    },
    # P816
    {
        "question": "What is the primary CATCODE (CCN) associated with project P816?",
        "options": ["A) 42132", "B) 74077", "C) 14320", "D) 11210", "E) 11025"],
        "answer": "A) 42132"
    },
    {
        "question": "How many additional DDGs are planned to be homeported at NS Rota by FY26 per project P816?",
        "options": ["A) 1", "B) 2", "C) 3", "D) 4", "E) 5"],
        "answer": "B) 2"
    },
    # P736
    {
        "question": "What is the installation location for project P736?",
        "options": ["A) NAVSTA Rota, Spain", "B) NAVSUPPACT Bahrain", "C) NAS Sigonella, Italy", "D) NAVSUPPACT Naples, Italy", "E) z/VARLOCS"],
        "answer": "C) NAS Sigonella, Italy"
    },
    {
        "question": "What is the overall capacity rating for project P736?",
        "options": ["A) 48", "B) 75", "C) 100", "D) 120", "E) 82"],
        "answer": "D) 120"
    },
    # P577
    {
        "question": "What is the current capacity rating (%) for the fitness facility addressed by project P577?",
        "options": ["A) 50%", "B) 60%", "C) 40%", "D) 70%", "E) 30%"],
        "answer": "C) 40%"
    },
    {
        "question": "What is the Region Urgency score for project P577?",
        "options": ["A) 1", "B) 2", "C) 3", "D) 4", "E) 0"],
        "answer": "D) 4"
    },
]


# ----------------------------------------------------
# EVAL FUNCTION  (parallelized + missed question tracking)
# ----------------------------------------------------

def eval_rag_chain_proj_query(my_rag_chain, q_num=15, verbose=False):
    """Evaluate RAG chain accuracy on a random sample of QA pairs.

    Returns
    -------
    accuracy : float
        Fraction of questions answered correctly.
    missed   : list[str]
        Question strings that were answered incorrectly (for miss-rate tracking).
    """
    q_a_pairs_rand = random.sample(q_a_pairs, q_num)

    def score_one(pair):
        question        = pair["question"] + " " + " ".join(pair["options"])
        expected_answer = pair["answer"]
        response        = my_rag_chain.invoke(question)
        correct         = expected_answer == response

        if verbose:
            print(f"Question: {question}")
            print(f"Correct Response:  {expected_answer}")
            max_len = 3 * len(expected_answer)
            print(f"Pipeline Response: {response[:max_len] if max_len < len(response) else response}\n")

        return {
            "question": pair["question"],   # short key for miss tracking
            "correct":  correct,
        }

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(score_one, q_a_pairs_rand))

    correct_count = sum(r["correct"] for r in results)
    missed        = [r["question"] for r in results if not r["correct"]]
    accuracy      = correct_count / len(results)

    if verbose:
        print(f"Pipeline accuracy: {accuracy:.02%}")

    return accuracy, missed

# ----------------------------------------------------
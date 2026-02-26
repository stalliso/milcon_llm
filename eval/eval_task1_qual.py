# RAG components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LangChain tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# LLM
from langchain_openai import ChatOpenAI

# ----------------------------------------------------
# DO NOT MODIFY THIS SCRIPT
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

# Function to evaluate RAG chain accuracy
def eval_rag_chain_proj_query(my_rag_chain):

    q_a_pairs = [
    {
        "question": "Explain the mission, scope and impact of RM15-0946.",
        "answer": '''
                    Mission: Facility provides housing for 344 junior enlisted personnel 
                    (E1-E4) in 172 two-person rooms. In addition, Building 266 houses Naval 
                    Security Forces and the Installation EOC.
                    Scope: This project repairs the HVAC system, electrical, plumbing, fire 
                    protection system, electronic entry control locks, doors, windows, fixtures, 
                    conveying equipment, and interior & exterior finishes in Bachelor Enlisted 
                    Quarters (BEQ) Bldg. 266. Project will eliminate current Poor condition of 
                    76 by upgrading the following Master System deficiencies: C30 Interior Finishes,
                    D10 Conveying, D20 Plumbing, D30 HVAC, D40 Fire Protection, and D50 Electrical.
                    Impact: Without this project, junior enlisted personnel will continue to 
                    live in deteriorated housing with major facility deficiencies. If left 
                    unchecked, the degradation could force the installation to rent space in 
                    off-base hotels or apartments at a cost of approximately $5.22M/year 
                    (not including transportation, security, or other associated costs).
                    '''
    },
    {
        "question": "Where is RM16-0799 located, how much is the current working estimate (CWE) and who is the lead proponent?",
        "answer": "Location: NAS Sigonella, Italy. CWE: $11.602M. Lead Proponent: Naval Air Enterprise (NAE)"
    },
    {
        "question": "Provide the ",
        "options": [
            "A) Blink Dog",
            "B) Ancient Bronze Dragon",
            "C) Tarrasque",
            "D) Blood Hawk"
        ],
        "answer": "D) Blood Hawk"
    },
    {
        "question": "What kind of elf has a keen mind?",
        "options": [
            "A) Wood Elf",
            "B) High Elf",
            "C) Dark Elf",
            "D) Elf-vis"
        ],
        "answer": "B) High Elf"
    },
        {
        "question": "Elves reaches adulthood around what age?",
        "options": [
            "A) 25",
            "B) 10,000",
            "C) 100",
            "D) 3.14159"
        ],
        "answer": "C) 100"
    },
    {
        "question": "Humans tend toward what alignment?",
        "options": [
            "A) Chaotic good",
            "B) Lawful evil",
            "C) Lawful good",
            "D) No particular alignment"
        ],
        "answer": "D) No particular alignment"
    },
    {
        "question": "What is Multiclassing?",
        "options": [
            "A) A way to gain levels in multiple classes.",
            "B) A method for classifying multiple things.",
            "C) A complex mathematical operation.",
            "D) A way to school your enemies."
        ],
        "answer": "A) A way to gain levels in multiple classes."
    },
    {
        "question": "What is a character's Proficiency Bonus based on?",
        "options": [
            "A) Strength score",
            "B) Total character level",
            "C) Hit Dice size",
            "D) Movement speed"
        ],
        "answer": "B) Total character level"
    },
    {
        "question": "Can you expend inspiration on an ability check?",
        "options": [
            "A) No",
            "B) Yes",
        ],
        "answer": "B) Yes"
    },
    {
        "question": "How much does it cost to silver a single weapon?",
        "options": [
            "A) 500 ep",
            "B) 25 pp",
            "C) 100 gp",
            "D) 10 cp"
        ],
        "answer": "C) 100 gp"
    },
    {
        "question": "When using a portable ram, what is the bonus on the Strength check?",
        "options": [
            "A) Over 9000",
            "B) There is no bonus",
            "C) -3",
            "D) +4"
        ],
        "answer": "D) +4"
    },
    {
        "question": "Which ability allows a character to see invisible creatures and objects?",
        "options": [
            "A) Truesight",
            "B) Heavy obscuration",
            "C) Spyglass",
            "D) Darkvision"
        ],
        "answer": "A) Truesight"
    },
    {
        "question": "What is the average weight of a Dwarf who is a level 20 Barbarian?",
        "options": [
            "A) 100 pounds",
            "B) 125 pounds",
            "C) 150 pounds",
            "D) 200 pounds"
        ],
        "answer": "C) 150 pounds"
    },
    {
        "question": "What is 'Stroke of Luck'?",
        "options": [
            "A) A high-level Rogue class ability",
            "B) An elusive monster",
            "C) A boring Wizard spell",
            "D) A halfing feat"
        ],
        "answer": "A) A high-level Rogue class ability"
    },
        {
        "question": "At level 1, which class is not proficient with light crossbows?",
        "options": [
            "A) Sorcerer",
            "B) Wizard",
            "C) Fighter",
            "D) Cleric"
        ],
        "answer": "D) Cleric"
    },
]

    # Counts correct answers to questions
    # Requires that answers match exactly
    correct = 0
    for pair in q_a_pairs:
        question = pair["question"] + " " + " ".join(pair["options"])
        expected_answer = pair["answer"]
        response = my_rag_chain.invoke(question)

        # Record correct answers
        if expected_answer == response:
            correct += 1
        
        print(f"Question: {question}")
        print(f"Correct Response:  {expected_answer}")
        # Don't print full response if it's longer than expected
        max_response_length = 3 * len(expected_answer)
        if max_response_length < len(response):
            print(f"Pipeline Response: {response[:max_response_length]}\n")
        else:
            print(f"Pipeline Response: {response}\n")

    accuracy = correct / len(q_a_pairs)

    print(f"Pipeline accuracy: {accuracy:.02%}")


# ----------------------------------------------------
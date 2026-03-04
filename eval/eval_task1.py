# RAG components
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LangChain tools
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

# LLM
from langchain_openai import ChatOpenAI

import random

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
def eval_rag_chain_proj_query(my_rag_chain, q_num=10):

    q_a_pairs = [
    # NF18-1440 - Upgrade Aircraft Fire Rescue Station - Bldg 58
    {
        "question": "What is the installation for project NF18-1440?",
        "options": [
            "A) NAS Sigonella",
            "B) NAVSTA Rota SP",
            "C) NAVSUPPACT Bahrain",
            "D) NAVSUPPACT Naples",
            "E) Camp Lemonnier"
        ],
        "answer": "B) NAVSTA Rota SP"
    },
    {
        "question": "What is the Lead Proponent for project NF18-1440?",
        "options": [
            "A) NAVFAC",
            "B) NAE",
            "C) CNIC N3",
            "D) IWE",
            "E) SWE"
        ],
        "answer": "C) CNIC N3"
    },

    # NF20-0826 - Port Operations Warehouse
    {
        "question": "What is the COCOM for project NF20-0826?",
        "options": [
            "A) AFRICOM",
            "B) CENTCOM",
            "C) TRANSCOM",
            "D) EUCOM",
            "E) INDOPACOM"
        ],
        "answer": "D) EUCOM"
    },
    {
        "question": "What is the Project CCN for NF20-0826?",
        "options": [
            "A) 14120",
            "B) 73025",
            "C) 44110",
            "D) 84109",
            "E) 11320"
        ],
        "answer": "C) 44110"
    },

    # NF23-0892 - Construct ARM-DEARM Safety Berm, CALA Improvements
    {
        "question": "What is the mission of the facility for project NF23-0892?",
        "options": [
            "A) Provides storage for ship power cables in proximity to berthing for Pier 1",
            "B) A Compensatory Barricade required as a site feature to accompany CALA and Arming/DeArming operations at NAVSTA ROTA in support of HSM79",
            "C) Provides aircraft parking apron to support EUCOM, CENTCOM, TRANSCOM, and AFRICOM missions",
            "D) Provides C5ISR for real-world operations and exercises to US Navy, Joint, and Coalition operating forces worldwide",
            "E) Single point of entry for vehicles and personnel at NSA Souda Bay"
        ],
        "answer": "B) A Compensatory Barricade required as a site feature to accompany CALA and Arming/DeArming operations at NAVSTA ROTA in support of HSM79"
    },
    {
        "question": "What is the Region Operational Cost score for project NF23-0892?",
        "options": [
            "A) 0",
            "B) 1",
            "C) 2",
            "D) 3",
            "E) 4"
        ],
        "answer": "C) 2"
    },

    # P001 - Provide Water Resiliency and Conservation Upgrades
    {
        "question": "What is the project scope for P001?",
        "options": [
            "A) Constructs a 4,000 SF warehouse to provide immediate storage for current and planned increase of shore cables at Pier 1",
            "B) Constructs a concrete T-Wall Barricade and HESCO Sand Container Barrier",
            "C) Builds a new golf course irrigation system consisting of new PVC piping, sprinkler heads, and pump station, and lines three existing depression areas to act as retention ponds",
            "D) Constructs 159,156 SM of lighted Aircraft Parking Apron for at least 22 aircraft",
            "E) Constructs a new pre-engineer building to house the Fire station and AIROPS departments"
        ],
        "answer": "C) Builds a new golf course irrigation system consisting of new PVC piping, sprinkler heads, and pump station, and lines three existing depression areas to act as retention ponds"
    },
    {
        "question": "What is the Region Severity Statement score for project P001?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "D) 4"
    },

    # P118 - Joint Entry Control Point and Vehicle Inspection Area
    {
        "question": "What is the mission of the facility for project P118?",
        "options": [
            "A) Provides storage for ship power cables in proximity to berthing for Pier 1",
            "B) Single point of entry for vehicles and personnel at NSA Souda Bay; secures and controls access to the installation",
            "C) Provide dockside utilities for ship service including shore power, communication system, potable water, and wastewater discharge",
            "D) Multi-story facility to house Naval Security Forces personnel, operations, and functions",
            "E) Provides aircraft parking apron to support EUCOM, CENTCOM, TRANSCOM, and AFRICOM missions"
        ],
        "answer": "B) Single point of entry for vehicles and personnel at NSA Souda Bay; secures and controls access to the installation"
    },
    {
        "question": "What is the Region Readiness Support score for project P118?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "D) 4"
    },

    # P119 - Aircraft Parking Apron (NAS II)
    {
        "question": "What is the impact if not provided for project P119?",
        "options": [
            "A) Shore power cables need to be transported by forklift which drives the need to be in proximity of Pier 1 berths",
            "B) The aircraft fire and rescue station will remain as is; fire fighting equipment will continue to be unnecessarily exposed to the elements",
            "C) The existing aircraft parking apron is not sufficient for daily operations and requires aircraft be redirected to other bases",
            "D) NAVSTA Rota will not comply with Mission Assurance Assessment Team recommendations",
            "E) CNE/CNA/C6F commanders will be forced to operate in an undersized, ill-configured and inadequately-capable MOC"
        ],
        "answer": "C) The existing aircraft parking apron is not sufficient for daily operations and requires aircraft be redirected to other bases"
    },
    {
        "question": "What is the Region Mission Alignment score for project P119?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "D) 4"
    },

    # P160 - Consolidated Port Operations Facilities
    {
        "question": "What is the installation for project P160?",
        "options": [
            "A) NAVSTA Rota SP",
            "B) NAS Sigonella",
            "C) NAVSUPPACT Naples",
            "D) NAVSUPPACT Souda Bay",
            "E) NAVSUPPACT Bahrain"
        ],
        "answer": "D) NAVSUPPACT Souda Bay"
    },
    {
        "question": "What is the Region Severity Statement score for project P160?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "C) 3"
    },

    # P187 - Shore to Ship Utilities (Mina Salman Pier)
    {
        "question": "What is the mission of the facility for project P187?",
        "options": [
            "A) Port Operations and Port Security including Oil Spill containment services, small craft maintenance and harbor patrol",
            "B) Provide dockside utilities for ship service including shore power, communication system, potable water, and discharge of wastewater at Mina Salman Pier",
            "C) Military Surface Deployment and Distribution Command requires adequate cargo staging area to temporarily store rolling stock",
            "D) Provides storage for ship power cables in proximity to berthing for Pier 1",
            "E) Multi-story facility to house Naval Security Forces personnel, operations, and functions"
        ],
        "answer": "B) Provide dockside utilities for ship service including shore power, communication system, potable water, and discharge of wastewater at Mina Salman Pier"
    },
    {
        "question": "What is the Lead Proponent for project P187?",
        "options": [
            "A) CNIC N3",
            "B) NAE",
            "C) NAVFAC",
            "D) IWE",
            "E) SWE"
        ],
        "answer": "E) SWE"
    },

    # P204 - Consolidated Security Facility
    {
        "question": "What is the Project CCN for P204?",
        "options": [
            "A) 14380",
            "B) 81320",
            "C) 73020",
            "D) 72111",
            "E) 44110"
        ],
        "answer": "C) 73020"
    },
    {
        "question": "What is the Region Mission Alignment score for project P204?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "C) 3"
    },

    # P221 - Aircraft Ready Fuels Storage at Isa Air Base
    {
        "question": "What is the project scope for P221?",
        "options": [
            "A) Constructs 20 MVA substation, comprehensive utility distribution systems and hotel stations to provide dedicated shore-to-ship utility services",
            "B) Constructs 33,350 barrel above ground vertical bulk storage tanks, two airfield refueler loading facilities with canopies, a canopy covered airfield refueler parking area, and two operational support facilities",
            "C) Installs a Clean Water Rinse System and all associated utility extensions and connections for rinsing aircraft upon return to the installation",
            "D) Constructs a new pre-engineer building to house the Fire station and AIROPS departments",
            "E) Constructs parking apron/taxiway, single bay A/C maintenance/wash hangar, rinse facility, sonobuoy storage, and admin/maintenance spaces"
        ],
        "answer": "B) Constructs 33,350 barrel above ground vertical bulk storage tanks, two airfield refueler loading facilities with canopies, a canopy covered airfield refueler parking area, and two operational support facilities"
    },
    {
        "question": "What is the Region Severity Statement score for project P221?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "C) 3"
    },

    # P222 - Unaccompanied Housing Phase 1
    {
        "question": "What is the mission of the facility for project P222?",
        "options": [
            "A) Provides critical nutritional sustainance for NSA Rota active military personnel",
            "B) Provide Aircraft ready fuels for the Aircrafts operating at Isa Air Base",
            "C) The facility will provide Mission Essential Unaccompanied Housing for personnel on rotational 6-12 month deployments",
            "D) Provides storage for furniture and personal items at Naval Station Rota",
            "E) Multi-story facility to house Naval Security Forces personnel, operations, and functions"
        ],
        "answer": "C) The facility will provide Mission Essential Unaccompanied Housing for personnel on rotational 6-12 month deployments"
    },
    {
        "question": "What is the Region Readiness Support score for project P222?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "C) 3"
    },

    # P223 - Flight Line Fire Station & Air Operations Facility
    {
        "question": "What is the RAC for project P223?",
        "options": [
            "A) I",
            "B) II",
            "C) III",
            "D) IV",
            "E) Not listed"
        ],
        "answer": "E) Not listed"
    },
    {
        "question": "What is the impact if not provided for project P223?",
        "options": [
            "A) Deficit in storage space will continue impacting the mission of Supply Ships Units operating in the area of responsibility",
            "B) The camp's personnel, equipment, facilities and airfield will be at risk of damage and destruction due to fire threats; the current facility cannot house the fire fighting force needed to adequately protect and serve the growing needs of the mission",
            "C) Failure to construct this fuel storage facility will limit the ability to provide a safe, efficient, and reliable fueling service",
            "D) Military personnel will continue to live in inadequate and unsafe living conditions, adversely impacting safety, morale, and readiness",
            "E) NSF will continue degraded mission capabilities in inadequately sized and deteriorating facilities"
        ],
        "answer": "B) The camp's personnel, equipment, facilities and airfield will be at risk of damage and destruction due to fire threats; the current facility cannot house the fire fighting force needed to adequately protect and serve the growing needs of the mission"
    },

    # P224 - P-8A Clear Water Rinse System (CWRS)
    {
        "question": "What is the mission of the facility for project P224?",
        "options": [
            "A) Directly supports both NATO and EDI requirement to improve and increase Anti-Submarine Warfare capability in the EUCOM area of operation",
            "B) The project constructs a Clear Water Rinse System for the removal of dust, salt, oil, and other corrosive contaminants from aircraft upon return to Isa Air Base",
            "C) Provide Aircraft ready fuels for the Aircrafts operating at Isa Air Base",
            "D) The facility will house the Flight Line Fire station and AIROPS departments which provide emergency response to US critical assets in the airfield",
            "E) Provides C5ISR for real-world operations and exercises to US Navy, Joint, and Coalition operating forces worldwide"
        ],
        "answer": "B) The project constructs a Clear Water Rinse System for the removal of dust, salt, oil, and other corrosive contaminants from aircraft upon return to Isa Air Base"
    },
    {
        "question": "What is the Region Mission Alignment score for project P224?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "C) 3"
    },

    # P309-EDI - P-8A Hangar & Airfield Pavement Improvement
    {
        "question": "What is the mission of the facility for project P309 EDI?",
        "options": [
            "A) Provides C5ISR for real-world operations and exercises to US Navy, Joint, and Coalition operating forces worldwide",
            "B) Directly supports both NATO and EDI requirement to improve and increase Anti-Submarine Warfare capability in the EUCOM area of operation",
            "C) Provide dockside utilities for ship service including shore power, communication system, potable water, and wastewater discharge",
            "D) Single point of entry for vehicles and personnel securing and controlling access to the installation",
            "E) Provide and maintain effective command and control for planning, execution, monitoring progress, and evaluating the progress of EURAFCENT Fleet operations"
        ],
        "answer": "B) Directly supports both NATO and EDI requirement to improve and increase Anti-Submarine Warfare capability in the EUCOM area of operation"
    },
    {
        "question": "What is the Region Mission Alignment score for project P309 EDI?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "D) 4"
    },

    # P309 - Satellite Communications Facility
    {
        "question": "What is the installation for project P309 Satellite Communications Facility?",
        "options": [
            "A) NAVSTA Rota SP",
            "B) NAS Sigonella",
            "C) NAVSUPPACT Bahrain",
            "D) NAVSUPPACT Naples",
            "E) Camp Lemonnier"
        ],
        "answer": "D) NAVSUPPACT Naples"
    },
    {
        "question": "What is the Lead Proponent for project P309 Satellite Communications Facility?",
        "options": [
            "A) CNIC N3",
            "B) NAVFAC",
            "C) NAE",
            "D) SWE",
            "E) IWE"
        ],
        "answer": "E) IWE"
    },

    # P314 - Naval Fleet Mission Operations Command and Control
    {
        "question": "What is the COCOM for project P314?",
        "options": [
            "A) CENTCOM",
            "B) AFRICOM",
            "C) TRANSCOM",
            "D) EUCOM/AFRICOM/C6F",
            "E) INDOPACOM"
        ],
        "answer": "D) EUCOM/AFRICOM/C6F"
    },
    {
        "question": "What is the Region Severity Statement score for project P314?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "D) 4"
    },

    # P328 - Dining Facility Replacement
    {
        "question": "What is the Project CCN for P328?",
        "options": [
            "A) 15310",
            "B) 71477",
            "C) 72210",
            "D) 14380",
            "E) 73020"
        ],
        "answer": "C) 72210"
    },
    {
        "question": "What is the Region Readiness Support score for project P328?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "C) 3"
    },

    # P334 - Electrical Power Plant
    {
        "question": "What is the mission of the facility for project P334?",
        "options": [
            "A) Provides storage for furniture and personal items at Naval Station Rota",
            "B) Project will provide reliable and continuous electrical power as necessary, provide redundancy in the electrical system where none exists, and significantly increase resiliency of the critical utility system onboard",
            "C) Provide and maintain effective command and control for planning, execution, monitoring progress, and evaluating the progress of EURAFCENT Fleet operations",
            "D) Provide dockside utilities for ship service including shore power, communication system, potable water, and wastewater discharge",
            "E) Provides C5ISR for real-world operations and exercises to US Navy, Joint, and Coalition operating forces worldwide"
        ],
        "answer": "B) Project will provide reliable and continuous electrical power as necessary, provide redundancy in the electrical system where none exists, and significantly increase resiliency of the critical utility system onboard"
    },
    {
        "question": "What is the Region Mission Alignment score for project P334?",
        "options": [
            "A) 1",
            "B) 2",
            "C) 3",
            "D) 4",
            "E) 0"
        ],
        "answer": "D) 4"
    },

    # P354 - Intermodal Operations Support
    {
        "question": "What is the mission of the facility for project P354?",
        "options": [
            "A) Facility provides critical nutritional sustainance for NSA Rota active military personnel",
            "B) NAVSTA Rota Housing is responsible for storing furniture, appliances, maintenance tools, supplies and other materials for family housing and bachelor quarter facilities",
            "C) Military Surface Deployment and Distribution Command requires adequate cargo staging area to temporarily store rolling stock, shipping containers, equipment, and airframes in support of intermodal operations",
            "D) Provides storage for ship power cables in proximity to berthing for Pier 1",
            "E) Provides aircraft parking apron to support EUCOM, CENTCOM, TRANSCOM, and AFRICOM missions"
        ],
        "answer": "C) Military Surface Deployment and Distribution Command requires adequate cargo staging area to temporarily store rolling stock, shipping containers, equipment, and airframes in support of intermodal operations"
    },
    {
        "question": "What is the COCOM for project P354?",
        "options": [
            "A) EUCOM",
            "B) AFRICOM",
            "C) CENTCOM",
            "D) TRANSCOM",
            "E) INDOPACOM"
        ],
        "answer": "D) TRANSCOM"
    },

    # P369 - Housing Warehouse - Distribution Center
    {
        "question": "What is the impact if not provided for project P369?",
        "options": [
            "A) Safe and adequate airfield infrastructure and supporting facilities needed to meet operational requirements will not be available",
            "B) Annual off-base leasing cost will continue to serve in lieu of adequate storage space; with the increase in mission expected with 2 additional FDNF ships crews and their dependants, demand will increase for storage of FH and UH material aboard NAVSTA Rota",
            "C) Failure to meet this electrical demand will result in substantial and direct negative mission impact, including load shed plans and rolling brownouts",
            "D) Deficit in storage space will continue impacting the mission of Supply Ships Units operating in the area of responsibility",
            "E) Without the facility, NSF will continue degraded mission capabilities in inadequately sized and deteriorating facilities"
        ],
        "answer": "B) Annual off-base leasing cost will continue to serve in lieu of adequate storage space; with the increase in mission expected with 2 additional FDNF ships crews and their dependants, demand will increase for storage of FH and UH material aboard NAVSTA Rota"
    },
    {
        "question": "What is the Project CCN for P369?",
        "options": [
            "A) 72210",
            "B) 15310",
            "C) 14380",
            "D) 71477",
            "E) 73020"
        ],
        "answer": "D) 71477"
    },
    # P503 - Magazine Expansion
    {
        "question": "What is the project scope for P503 Magazine Expansion at Camp Lemonnier Djibouti?",
        "options": [
            "A) Constructs two new earth-covered magazines classified for 45K Net Explosive Weight",
            "B) Constructs four new Type 'D' RC Box Magazines classified for 45K Net Explosive Weight, for a total of eight",
            "C) Constructs six new magazines with hardened bunkers rated for 60K Net Explosive Weight",
            "D) Expands existing magazines with additional storage bays for 30K Net Explosive Weight",
            "E) Constructs a single consolidated magazine facility with 90K total Net Explosive Weight capacity"
        ],
        "answer": "B) Constructs four new Type 'D' RC Box Magazines classified for 45K Net Explosive Weight, for a total of eight"
    },
    {
        "question": "What is the CWE for project P503 Magazine Expansion?",
        "options": [
            "A) $47,890K",
            "B) $82,400K",
            "C) $64,320K",
            "D) $94,670K",
            "E) $76,540K"
        ],
        "answer": "C) $64,320K"
    },

    # P506 - Aircraft Parking Apron Expansion Phase II
    {
        "question": "What is the impact if not provided for project P506 Aircraft Parking Apron Expansion Phase II?",
        "options": [
            "A) CLDJ will continue to have an ammunition storage deficit of 50%, creating life safety hazards",
            "B) Aircraft parking will require a NAVAIR waiver to violate minimum wing tip clearances, risking damage to aircraft especially given nearly 100% of missions are executed at night",
            "C) Sailors will continue to use a fitness facility that is undersized and not configured to provide adequate training",
            "D) Fire trucks will continue to be kept outside in extreme temperatures, limiting response capabilities",
            "E) Shore power will not be available for outboard ships during nested mooring, forcing them to AUX steam"
        ],
        "answer": "B) Aircraft parking will require a NAVAIR waiver to violate minimum wing tip clearances, risking damage to aircraft especially given nearly 100% of missions are executed at night"
    },
    {
        "question": "What is the Project CCN for P506 Aircraft Parking Apron Expansion Phase II?",
        "options": [
            "A) 14112",
            "B) 42122",
            "C) 72127",
            "D) 11320",
            "E) 21116"
        ],
        "answer": "D) 11320"
    },

    # P577 - Physical Fitness Training Center
    {
        "question": "What is the mission of the facility for project P577 Physical Fitness Training Center?",
        "options": [
            "A) Provide a comprehensive Physical Fitness Facility for all NS Rota demographics",
            "B) Provide applied instruction operational training space in support of Combat Systems trainers",
            "C) Provide fire protection and emergency response for aviation, motor vehicle, and structural incidents",
            "D) Provide a cafeteria-style dining facility for enlisted personnel at NAS Sigonella",
            "E) Provide aircraft maintenance and repair services for aeronautical components at NAVSTA Rota"
        ],
        "answer": "A) Provide a comprehensive Physical Fitness Facility for all NS Rota demographics"
    },
    {
        "question": "What is the overall capacity rating for the existing Physical Fitness Training Center facility associated with project P577?",
        "options": [
            "A) 28%",
            "B) 51%",
            "C) 65%",
            "D) 40%",
            "E) 18%"
        ],
        "answer": "D) 40%"
    },

    # P650 - Construct CIAT and ATET SCS Training Facility
    {
        "question": "What is the project scope for P650 Construct CIAT and ATET SCS Training Facility at NAVSTA Rota?",
        "options": [
            "A) Constructs a two-story 52,000 SF consolidated maintenance and administrative facility for AIMD operations",
            "B) Constructs a single-story 34K SF applied instruction operational training space supporting Combat Systems trainers, primarily CIAT and ATET, plus administrative office space",
            "C) Constructs a 90,000 SF Joint Use Communications and Core Data Center facility",
            "D) Constructs a 10K SF inert storage space and expands the NMC operations building",
            "E) Constructs an 82K SF indoor physical fitness facility with demolition of legacy fitness center"
        ],
        "answer": "B) Constructs a single-story 34K SF applied instruction operational training space supporting Combat Systems trainers, primarily CIAT and ATET, plus administrative office space"
    },
    {
        "question": "What is the COCOM for project P650 Construct CIAT and ATET SCS Training Facility?",
        "options": [
            "A) AFRICOM",
            "B) CENTCOM",
            "C) TRANSCOM",
            "D) INDOPACOM",
            "E) EUCOM"
        ],
        "answer": "E) EUCOM"
    },

    # P736 - Entry Control Point (NAS I)
    {
        "question": "What is the mission of the facility for project P736 Entry Control Point (NAS I) at NAS Sigonella?",
        "options": [
            "A) Provide fire protection and emergency rescue services for the airfield and over 1,053 real property assets",
            "B) Encompass the overall layout, infrastructure, and facilities to perform visitor processing, vehicle registration, ID checks, POV inspections, and commercial vehicle inspections",
            "C) Provide ordnance logistics including loading and unloading of ordnance onto aircraft and organizational-level maintenance",
            "D) Provide a cafeteria-style dining facility for enlisted personnel",
            "E) Provide adequate consolidated maintenance and storage space for AIMD operations"
        ],
        "answer": "B) Encompass the overall layout, infrastructure, and facilities to perform visitor processing, vehicle registration, ID checks, POV inspections, and commercial vehicle inspections"
    },
    {
        "question": "What is the Lead Proponent Severity Statement score for project P736 Entry Control Point (NAS I)?",
        "options": [
            "A) 4 - This project will eliminate serious life safety hazards",
            "B) 2 - Project improves base operations efficiency by increasing vehicle and pedestrian throughput while minimizing ATFP risk",
            "C) 1 - Minor impact on daily operations with adequate existing workarounds",
            "D) 3 - ATFP risk high for important primary gathering facilities; critical assets assessed as high AT/FP risk and red operational gap",
            "E) 0 - No documented severity"
        ],
        "answer": "D) 3 - ATFP risk high for important primary gathering facilities; critical assets assessed as high AT/FP risk and red operational gap"
    },

    # P738 - Combined Structural & Aircraft Fire Rescue Station (NAS II)
    {
        "question": "What is the impact if not provided for project P738 Combined Structural & Aircraft Fire Rescue Station (NAS II)?",
        "options": [
            "A) AIMD will continue to operate in inadequately sized and poorly configured space creating unsafe and inefficient work environments",
            "B) Response times have been calculated to take three times longer than the NFPA 1710 standard due to the 14,735 SF space deficit, jeopardizing fire protection for the airfield and over 1,053 real property assets",
            "C) Weapons operations will continue in facilities not optimally configured to support ordnance administration and handling",
            "D) CLDJ will continue to use approximately 708 beds that are non-compliant with UFC and NFPA",
            "E) Out-of-compliance entry control will leave personnel vulnerable to VBIED attacks"
        ],
        "answer": "B) Response times have been calculated to take three times longer than the NFPA 1710 standard due to the 14,735 SF space deficit, jeopardizing fire protection for the airfield and over 1,053 real property assets"
    },
    {
        "question": "What is the PRV-Weighted Target Investment Zone for the existing Aircraft Fire Rescue Station facility associated with project P738?",
        "options": [
            "A) 57",
            "B) 39",
            "C) 68",
            "D) 80",
            "E) 98"
        ],
        "answer": "E) 98"
    },

    # P740 - Consolidated AIMD/GSE Shop (NAS II)
    {
        "question": "What is the mission of the facility for project P740 Consolidated AIMD/GSE Shop (NAS II)?",
        "options": [
            "A) Provide operations space for Airframe Intermediate Maintenance Division to conduct maintenance, testing, and repair on aeronautical components supporting EUCOM, AFRICOM, and CENTCOM",
            "B) Provide fire protection and emergency response for aviation, motor vehicle, and structural incidents at NAS Sigonella",
            "C) Provide physical infrastructure to support the global DoD Joint Information Enterprise platform",
            "D) Provide ordnance logistics including loading and unloading of ordnance onto aircraft at NAS Sigonella",
            "E) Provide a combined administrative and combat systems training space for sailors at NAS Sigonella"
        ],
        "answer": "A) Provide operations space for Airframe Intermediate Maintenance Division to conduct maintenance, testing, and repair on aeronautical components supporting EUCOM, AFRICOM, and CENTCOM"
    },
    {
        "question": "What is the overall capacity rating for the AIMD/GSE Shop facility associated with project P740?",
        "options": [
            "A) 57%",
            "B) 28%",
            "C) 48%",
            "D) 100%",
            "E) 240%"
        ],
        "answer": "E) 240%"
    },

    # P741 - Ordnance Operations Facility (NAS II)
    {
        "question": "What is the mission of the facility for project P741 Ordnance Operations Facility (NAS II)?",
        "options": [
            "A) Provide an applied instruction facility for Combat Systems trainers in support of DESRON personnel",
            "B) The Naval Munitions Command Sigonella Detachment provides ordnance logistics including loading and unloading of ordnance onto aircraft, organizational-level maintenance, and storing weapons and ordnance for tenant missions throughout the European Theater",
            "C) Provide operations space for Airframe Intermediate Maintenance Division to conduct maintenance on aeronautical components",
            "D) Provide physical infrastructure to support the global DoD Joint Information Enterprise platform in the EUCOM AOR",
            "E) Provide a comprehensive fitness facility for all NAS Sigonella demographics"
        ],
        "answer": "B) The Naval Munitions Command Sigonella Detachment provides ordnance logistics including loading and unloading of ordnance onto aircraft, organizational-level maintenance, and storing weapons and ordnance for tenant missions throughout the European Theater"
    },
    {
        "question": "What is the Region Readiness Support score and description for project P741 Ordnance Operations Facility (NAS II)?",
        "options": [
            "A) Score 4 - Project eliminates documented serious life safety hazards (RAC I)",
            "B) Score 2 - Project addresses immediate gaps in ordnance operations and infrastructure necessary to meet NAVEUR ordnance operations requirements to effectively respond to and sustain rearm/resupply for OPLANs",
            "C) Score 3 - NAS Sigonella serves as a key ordnance storage location in the Mediterranean and designated as a secondary ordnance load out site and for pre-positioned war reserve ordnance storage",
            "D) Score 1 - Supports occasional short-term transient joint operations",
            "E) Score 4 - Project is critical to Fleet/COCOM operations and directly supports Navy QOL initiatives"
        ],
        "answer": "C) Score 3 - NAS Sigonella serves as a key ordnance storage location in the Mediterranean and designated as a secondary ordnance load out site and for pre-positioned war reserve ordnance storage"
    },

    # P742 - Galley (NAS II)
    {
        "question": "What is the project scope for P742 Galley (NAS II) at NAS Sigonella?",
        "options": [
            "A) Constructs a 35,542 SF one-story dining facility to provide additional capacity for a camp population of over 5,000",
            "B) Constructs a larger Galley to meet existing and increased mission requirements and demolishes the existing non-AT compliant, undersized, and inadequate dining facility",
            "C) Constructs a new 82K SF indoor physical fitness facility and demolishes the legacy 1950s facilities",
            "D) Constructs a two-story combined structural aircraft fire rescue station including apparatus stalls and dormitory rooms",
            "E) Constructs a consolidated 52,000 SF maintenance and administrative facility for AIMD operations"
        ],
        "answer": "B) Constructs a larger Galley to meet existing and increased mission requirements and demolishes the existing non-AT compliant, undersized, and inadequate dining facility"
    },
    {
        "question": "What deficiency codes are documented for the existing Galley facility (Building 533) associated with project P742?",
        "options": [
            "A) W7 ATFP non-compliance, X3 Functional/Space Criteria, Z5 Inadequate Capacity",
            "B) W2 Seismic non-compliance, W7 ATFP non-compliance, and X2 Building interior configuration inadequacy",
            "C) W5 Fire Code non-compliance, X3 Functional/Space Criteria, Y1 Location/Siting",
            "D) Y3 Site Characteristics, Z0 Alarm System inadequacy, Z5 Facility Components inadequacy",
            "E) X2 Building interior configuration, Y2 Flood Plain incompatibility, Z9 Inadequate capacity"
        ],
        "answer": "B) W2 Seismic non-compliance, W7 ATFP non-compliance, and X2 Building interior configuration inadequacy"
    },

    # P816 - Expand NMC Ops Building, Construct Inert Storage Warehouse
    {
        "question": "What is the impact if not provided for project P816 Expand NMC Ops Building, Construct Inert Storage Warehouse at NAVSTA Rota?",
        "options": [
            "A) LCACs will continue to not use the existing ramp due to operational risks and will conduct all loading and unloading on the nearby beach",
            "B) NMCLANT Det. Rota will not have the capability to store approximately 500 assets including inert VLS equipment, missile canister adaptors, missile cell covers, and ordnance handling equipment valued over $2M that support the growing FDNF-E FOS mission",
            "C) Out-board ships in nested position will have to AUX steam more frequently, putting unnecessary stress on personnel and equipment",
            "D) Combat training requirements will go unmet as ATET and CIAT trainers cannot be installed in inadequate existing facilities",
            "E) Sailors will continue to use a fitness facility with a 40% capacity rating and no off-base alternatives"
        ],
        "answer": "B) NMCLANT Det. Rota will not have the capability to store approximately 500 assets including inert VLS equipment, missile canister adaptors, missile cell covers, and ordnance handling equipment valued over $2M that support the growing FDNF-E FOS mission"
    },
    {
        "question": "What is the Project CCN and CWE for project P816 Expand NMC Ops Building, Construct Inert Storage Warehouse?",
        "options": [
            "A) CCN 15966, CWE $7,370K",
            "B) CCN 74044, CWE $62,520K",
            "C) CCN 42132, CWE $27,700K",
            "D) CCN 81320, CWE $13,350K",
            "E) CCN 14320, CWE $12,960K"
        ],
        "answer": "C) CCN 42132, CWE $27,700K"
    },

    # P827 - LCAC Ramp Expansion
    {
        "question": "What is the mission of the facility for project P827 LCAC Ramp Expansion at NAVSTA Rota?",
        "options": [
            "A) Provide aircraft parking apron to support EUCOM, TRANSCOM, and USAF operations at NAVSTA Rota",
            "B) The agricultural wash down and inspection process is a requirement for re-entry into the United States per USDA and Customs and Border Patrol to ensure no contraband or foreign invasive species are introduced",
            "C) Provide adequate shore power that is sufficiently transformed and converted to support USN Forward Deployed Naval Forces homeported at Rota",
            "D) Provide consolidated inert storage space for NMCLANT Det. Rota to support growing FDNF-E FOS mission",
            "E) Provide aircraft maintenance capabilities for rotating squadron units at Isa Air Base"
        ],
        "answer": "B) The agricultural wash down and inspection process is a requirement for re-entry into the United States per USDA and Customs and Border Patrol to ensure no contraband or foreign invasive species are introduced"
    },
    {
        "question": "What is the Lead Proponent Mission Alignment score and description for project P827 LCAC Ramp Expansion?",
        "options": [
            "A) Score 4 - Failure to execute this project results in critical warfighter capability gap and mission degradation; the LCAC ramp cannot be used without required expansion",
            "B) Score 2 - Prime mission accomplished with moderate workarounds; NMC Det. Rota requires inert storage to support growing mission",
            "C) Score 3 - Project expands LCAC ramp so it can be safely and efficiently used for wash down operations; addresses capability gap WO-07 in the Waterfront Operations Area Development Plan",
            "D) Score 1 - Supports occasional short-term joint operations with significant workarounds",
            "E) Score 4 - Project directly supports SECDEF/SECNAV/CNO directed basing decisions for increased mission"
        ],
        "answer": "C) Score 3 - Project expands LCAC ramp so it can be safely and efficiently used for wash down operations; addresses capability gap WO-07 in the Waterfront Operations Area Development Plan"
    },

    # P912 - Shore to Ship Electrical Substation
    {
        "question": "What is the project scope for P912 Shore to Ship Electrical Substation at NAVSTA Rota?",
        "options": [
            "A) Expands the existing LCAC ramp and constructs a laydown/staging area at the NAVSTA Rota waterfront",
            "B) Installs two shore power substations in Berths #3 and #5 at Pier 1, including circuit breakers, dedicated feeders, and SF6 circuit breakers with new controls at the main installation switchgear",
            "C) Constructs 10K SF inert storage space with site prep, parking, and access, demolishing several legacy facilities",
            "D) Constructs a parking apron adjacent to the new PAX terminal to support TRANSCOM and USAF throughput",
            "E) Installs a new underground electrical distribution system to support all four DDG berths at Pier 1"
        ],
        "answer": "B) Installs two shore power substations in Berths #3 and #5 at Pier 1, including circuit breakers, dedicated feeders, and SF6 circuit breakers with new controls at the main installation switchgear"
    },
    {
        "question": "What is the Region Mission Alignment score and description for project P912 Shore to Ship Electrical Substation?",
        "options": [
            "A) Score 3 - The existing ECP does not comply with UFC security standards; this project strengthens resilience of critical assets",
            "B) Score 2 - Prime mission accomplished with moderate workarounds; NMC Ops building requires expansion to accommodate growth",
            "C) Score 4 - Project directly supports SECDEF/SECNAV/CNO directed basing decision for increased missions; will provide additional shore power substations for DDG vessels to allow shore power for all in-port ships and avoid AUX Steam",
            "D) Score 4 - Contributes to closing capability gap identified in an approved Enterprise or Theater planning document",
            "E) Score 3 - Project contributes to closure of a NAVSTA Rota IDP capability gap identified during Installation Integration Group meetings"
        ],
        "answer": "C) Score 4 - Project directly supports SECDEF/SECNAV/CNO directed basing decision for increased missions; will provide additional shore power substations for DDG vessels to allow shore power for all in-port ships and avoid AUX Steam"
    },

    # P922 - Unaccompanied Housing Ph 3, Austere Standard
    {
        "question": "What is the project scope for P922 Unaccompanied Housing Ph 3, Austere Standard at Camp Lemonnier Djibouti?",
        "options": [
            "A) Constructs 52,800 SF 5-story barracks with 250 (2+0) rooms to accommodate 500 E1-E6 personnel",
            "B) Constructs 88,049 SF 5-story barracks consisting of 113 (4+0) rooms with centralized bathrooms to accommodate 708 E1-E6 personnel",
            "C) Constructs 76,540 SF 4-story barracks with 200 (2+0) rooms to accommodate 600 E1-E6 personnel",
            "D) Constructs three 5-story barracks buildings totaling 250,000 SF to accommodate 2,000 personnel",
            "E) Constructs a 40,000 SF single-story austere barracks with individual room HVAC units"
        ],
        "answer": "B) Constructs 88,049 SF 5-story barracks consisting of 113 (4+0) rooms with centralized bathrooms to accommodate 708 E1-E6 personnel"
    },
    {
        "question": "What is the RAC rating and the Region Severity Statement description for project P922 Unaccompanied Housing Ph 3?",
        "options": [
            "A) RAC II - Three out of seven existing fire station truck bays are inoperable; facility has multiple condition deficiencies",
            "B) RAC I - There are 1,416 beds throughout the base assessed RAC 1 for lacking code compliant fire protection systems; tents and CLUs lack wall hardening to meet AT/FP compliance",
            "C) RAC II - Facility is 47 years old and configuration is ineffective in supporting airfield, structural, and ancillary service mission sets",
            "D) RAC I - The existing galley operates 22 out of 24 hours and cannot adequately serve the 5,115 person steady state population",
            "E) RAC III - The current ECP does not comply with UFC security standards and poses moderate risk"
        ],
        "answer": "B) RAC I - There are 1,416 beds throughout the base assessed RAC 1 for lacking code compliant fire protection systems; tents and CLUs lack wall hardening to meet AT/FP compliance"
    },

    # P923 - Unaccompanied Housing Ph 4, Austere Standard
    {
        "question": "What is the impact if not provided for project P923 Unaccompanied Housing Ph 4, Austere Standard at Camp Lemonnier Djibouti?",
        "options": [
            "A) CLDJ will continue to use approximately 708 beds that are non-compliant with UFC and NFPA; energy consumption will continue to be high",
            "B) CLDJ will continue to house sailors in tents and CLUs that are non-compliant with UFC and NFPA; AFRICOM will be unable to meet OPLAN requirements and energy consumption will remain high",
            "C) AirOps cannot adequately support COCOMs current and projected air/sea RSOI requirements; CLDJ has a 98K SF cargo/passenger deficit",
            "D) Without this project the current galley will continue to experience overcrowded conditions operating 22 out of 24 hours",
            "E) The ammunition storage deficit will continue, requiring personnel to store munitions on aprons and aircraft"
        ],
        "answer": "B) CLDJ will continue to house sailors in tents and CLUs that are non-compliant with UFC and NFPA; AFRICOM will be unable to meet OPLAN requirements and energy consumption will remain high"
    },
    {
        "question": "What is the CWE and Lead Proponent Operational Cost score for project P923 Unaccompanied Housing Ph 4?",
        "options": [
            "A) CWE $76,540K, Score 4 - Project results in estimated 20% reduction in energy consumption versus CLUs",
            "B) CWE $67,920K, Score 0 - No significant cost savings documented",
            "C) CWE $82,400K, Score 1 - Project results in estimated 22% reduction in energy consumption; standard permanent BEQ expected to consume 30% less than equivalent CLUs",
            "D) CWE $64,320K, Score 2 - Moderate cost savings demonstrated through reduced maintenance costs",
            "E) CWE $94,670K, Score 1 - Project eliminates NAVAIR wingtip clearance waivers, reducing operational costs"
        ],
        "answer": "C) CWE $82,400K, Score 1 - Project results in estimated 22% reduction in energy consumption; standard permanent BEQ expected to consume 30% less than equivalent CLUs"
    },

    # P930 - Galley II
    {
        "question": "What is the mission of the facility for project P930 Galley II at Camp Lemonnier Djibouti?",
        "options": [
            "A) A dining facility for enlisted personnel providing cafeteria-style feeding of short order and regular meals at NAS Sigonella",
            "B) Facility provides critical nutritional sustainment for the base population, which is currently over 5,000 personnel",
            "C) Provide comprehensive Quality of Service dining for all NAVSTA Rota demographics",
            "D) Provide a combined galley and MWR facility for AFRICOM theater personnel",
            "E) Provide contingency food service operations for SOCOM and special operations forces"
        ],
        "answer": "B) Facility provides critical nutritional sustainment for the base population, which is currently over 5,000 personnel"
    },
    {
        "question": "What is the Region Urgency Statement score and description for project P930 Galley II?",
        "options": [
            "A) Score 2 - Resourcing required in BY+3 to meet projected mission growth (P-8A, Triton, etc.)",
            "B) Score 3 - Resourcing required in BY+2 to meet existing permanent party requirements",
            "C) Score 4 - Resourcing required in BY to meet existing force structure growth arriving and existing ops permanent party requirements; original Galley was sized for 2,200 people but current camp population is almost double and projected to double again by 2025",
            "D) Score 4 - Resourcing required in BY or sooner to meet IOC/FOC for 2 additional DDGs assigned to NS Rota by 2026",
            "E) Score 3 - Project funding required in current FY to avoid critical warfighter capability gap"
        ],
        "answer": "C) Score 4 - Resourcing required in BY to meet existing force structure growth arriving and existing ops permanent party requirements; original Galley was sized for 2,200 people but current camp population is almost double and projected to double again by 2025"
    },

    # P942 - Combined Air Cargo/Passenger Terminal
    {
        "question": "What is the mission of the facility for project P942 Combined Air Cargo/Passenger Terminal at Camp Lemonnier Djibouti?",
        "options": [
            "A) Provide physical infrastructure to support the global DoD Joint Information Enterprise platform in the AFRICOM AOR",
            "B) Provide 24/7 air operations and logistic support center in direct support of USAFRICOM, USCENTCOM, USSOCOM, USTRANSCOM, and other Government agencies in the East Africa area of operation",
            "C) Provide critical nutritional sustainment for the base population of over 5,000 personnel at CLDJ",
            "D) Provide centralized passenger processing in support of rotational forces at Camp Lemonnier",
            "E) Provide strategic airlift reception and staging for humanitarian assistance engagements"
        ],
        "answer": "B) Provide 24/7 air operations and logistic support center in direct support of USAFRICOM, USCENTCOM, USSOCOM, USTRANSCOM, and other Government agencies in the East Africa area of operation"
    },
    {
        "question": "What is the overall capacity rating for the existing Air Operations Complex associated with project P942?",
        "options": [
            "A) 46%",
            "B) 28%",
            "C) 77%",
            "D) 11%",
            "E) 65%"
        ],
        "answer": "D) 11%"
    },

    # P979 - Joint Use Communications Facility & Data Center
    {
        "question": "What is the project scope for P979 Joint Use Communications Facility & Data Center at NAVSUPPACT Bahrain?",
        "options": [
            "A) Constructs a 34K SF applied instruction operational training space for Combat Systems trainers at NSA Bahrain",
            "B) Constructs 90,000 SF Joint Use Communications & Core Data Center Facility at NSA II to provide secure, redundant, and resilient C5I infrastructure for trans-regional JIE C5I and COOP requirements; also reconfigures existing NCTS Data Center and upgrades communications and electrical infrastructure",
            "C) Constructs a 52,000 SF consolidated maintenance and administrative facility for AIMD operations at NSA Bahrain",
            "D) Constructs a 50,000 SF multipurpose logistics facility to support contingency operations in the CENTCOM AOR",
            "E) Constructs a new communications node to provide redundant connectivity between NSA Bahrain and NAVCENT headquarters"
        ],
        "answer": "B) Constructs 90,000 SF Joint Use Communications & Core Data Center Facility at NSA II to provide secure, redundant, and resilient C5I infrastructure for trans-regional JIE C5I and COOP requirements; also reconfigures existing NCTS Data Center and upgrades communications and electrical infrastructure"
    },
    {
        "question": "What is the Lead Proponent Mission Alignment score and description for project P979 Joint Use Communications Facility & Data Center?",
        "options": [
            "A) Score 3 - Project directly supports EUCOM operations and logistics sustainment in the AOR",
            "B) Score 4 - Limited IT services impacts DoD entities in CENTCOM AOR and beyond; ability to support operating forces throughout the battlespace is constrained, potentially impacting intelligence dissemination and forcing a combatant commander to delay operations or accept increased risk",
            "C) Score 3 - Fully meets BFR with a current capacity rating of 70%; consolidation will relocate critical C5I infrastructure from inadequate temporary facilities",
            "D) Score 4 - Bahrain is the only enduring installation in the CENTCOM AOR and provides the only enduring C5I/Data Center facility in the AOR",
            "E) Score 2 - Moderate impact on operations; workarounds are available but inefficient"
        ],
        "answer": "D) Score 4 - Bahrain is the only enduring installation in the CENTCOM AOR and provides the only enduring C5I/Data Center facility in the AOR"
    },

    # P998 - Construct Aircraft Maintenance Hangar
    {
        "question": "What is the mission of the facility for project P998 Construct Aircraft Maintenance Hangar at NAVSUPPACT Bahrain?",
        "options": [
            "A) Provide maintenance to air squadron units rotating through Isa AB approximately every six months and for aviation detachments tied to planned deployments of the Littoral Combat Ship to Bahrain, conducting crisis response and contingency operations in the CENTCOM AOR",
            "B) Provide aircraft maintenance and component repair services for aeronautical components in support of EUCOM, AFRICOM, and CENTCOM areas of operations",
            "C) Provide a pre-engineered maintenance hangar for P-8A Poseidon and Triton aircraft assigned to NSA Bahrain",
            "D) Provide consolidated GSE maintenance and administrative space for NAVCENT aviation assets",
            "E) Provide temporary aircraft storage and maintenance for transiting aircraft in the CENTCOM AOR"
        ],
        "answer": "A) Provide maintenance to air squadron units rotating through Isa AB approximately every six months and for aviation detachments tied to planned deployments of the Littoral Combat Ship to Bahrain, conducting crisis response and contingency operations in the CENTCOM AOR"
    },
    {
        "question": "What is the Region Severity Statement score and description for project P998 Construct Aircraft Maintenance Hangar?",
        "options": [
            "A) Score 4 - Primary mission accomplished with extreme workarounds",
            "B) Score 2 - The engine bays are undersized for some trucks and three of seven existing bays are inoperable",
            "C) Score 3 - Continued degradation of the only maintenance site due to age and extreme environmental conditions requires a new facility to ensure critical aircraft maintenance can be conducted on more than one assigned asset at a time; currently at 100% capacity for CMV-22 Osprey and MQ-9 Reaper missions",
            "D) Score 3 - Existing facility cannot support contingency operations per the 2020 Region Master Plan Gap Solutions",
            "E) Score 2 - Primary mission accomplished with moderate workarounds; simultaneous multi-aircraft maintenance is constrained"
        ],
        "answer": "C) Score 3 - Continued degradation of the only maintenance site due to age and extreme environmental conditions requires a new facility to ensure critical aircraft maintenance can be conducted on more than one assigned asset at a time; currently at 100% capacity for CMV-22 Osprey and MQ-9 Reaper missions"
    },

    # P1028 - Provide Parking Apron in support of new PAX
    {
        "question": "What is the project scope for P1028 Provide Parking Apron in support of new PAX at NAVSTA Rota?",
        "options": [
            "A) Constructs 159,517 SF aircraft parking apron and 5,667 SF taxiway in the Djiboutian Notch for 747s and C5s",
            "B) Expands the existing LCAC ramp and constructs a laydown/staging area at the NAVSTA Rota waterfront",
            "C) Provides a parking apron adjacent to the new PAX terminal; once the legacy PAX terminal is demolished as part of the enacted P695 MCON, the site will be reconstituted as aircraft ramp space for TRANSCOM and USAF throughput supporting two wide-body aircraft",
            "D) Constructs a new aircraft parking apron to support all FDNF DDG homeported aircraft at NAVSTA Rota",
            "E) Provides a dedicated ramp for commercial and government charter aircraft supporting rotational troop movements"
        ],
        "answer": "C) Provides a parking apron adjacent to the new PAX terminal; once the legacy PAX terminal is demolished as part of the enacted P695 MCON, the site will be reconstituted as aircraft ramp space for TRANSCOM and USAF throughput supporting two wide-body aircraft"
    },
    {
        "question": "What is the CWE and COCOM for project P1028 Provide Parking Apron in support of new PAX?",
        "options": [
            "A) CWE $94,670K, COCOM AFRICOM",
            "B) CWE $7,370K, COCOM EUCOM",
            "C) CWE $13,350K, COCOM EUCOM",
            "D) CWE $40,530K, COCOM EUCOM and TRANSCOM",
            "E) CWE $27,700K, COCOM EUCOM"
        ],
        "answer": "D) CWE $40,530K, COCOM EUCOM and TRANSCOM"
    },

    # P1121 - EDI: Prepositioned Logistics Facility
    {
        "question": "What is the mission of the facility for project P1121 EDI Prepositioned Logistics Facility at VARLOCs?",
        "options": [
            "A) Provide a mine countermeasure (MCM) capability and resilient logistics infrastructure at VARLOCs to allow for rapid response to contingency operations and sustain joint operations",
            "B) Provide a multipurpose logistics facility to support NAVEUR supply chain operations in the Mediterranean",
            "C) Establish forward-positioned ammunition storage for NATO contingency operations in Northern Europe",
            "D) Provide artic EOD capability storage and administrative support space for NECC operations",
            "E) Provide centralized warehousing for pre-positioned war reserve material in the EUCOM AOR"
        ],
        "answer": "A) Provide a mine countermeasure (MCM) capability and resilient logistics infrastructure at VARLOCs to allow for rapid response to contingency operations and sustain joint operations"
    },
    {
        "question": "What is the Lead Proponent Urgency Statement score and description for project P1121 EDI Prepositioned Logistics Facility?",
        "options": [
            "A) Score 3 - BY requirement; inert material shortcomings already being realized in current CY",
            "B) Score 2 - Resourcing required in BY+3 to meet projected mission growth",
            "C) Score 4 - Mission Need Date is submitted at 2029; project will require investment in BY or BY+1 to realistically meet expected BOD date of late 2028",
            "D) Score 4 - Failure to execute during this BY will result in delayed response for critical expeditionary capabilities in the North, limiting freedom of movement and degrading mission readiness",
            "E) Score 3 - Project is needed in BY+2 to support NAVEUR logistics sustainment operations"
        ],
        "answer": "D) Score 4 - Failure to execute during this BY will result in delayed response for critical expeditionary capabilities in the North, limiting freedom of movement and degrading mission readiness"
    },
    # P1125 - Multi-Purpose Logistics and Expeditionary Facility
    {
        "question": "What is the mission of the facility for project P1125?",
        "options": [
            "A) Facility provides unaccompanied housing for expeditionary forces stationed at NAVSTA Rota",
            "B) Facility establishes multipurpose logistics and expeditionary facility that supports Commander, Naval Forces Europe, and supported units, including TASW/ISR mission capabilities and arctic warfare training",
            "C) Facility provides energy resilience and cybersecurity for critical utility loads at NAVSTA Rota",
            "D) Facility provides operational storage and vehicle maintenance for CENTCOM AOR forces",
            "E) Facility provides multipurpose cargo staging for Military Surface Deployment and Distribution Command"
        ],
        "answer": "B) Facility establishes multipurpose logistics and expeditionary facility that supports Commander, Naval Forces Europe, and supported units, including TASW/ISR mission capabilities and arctic warfare training"
    },
    {
        "question": "What is the CWE and Project CCN for project P1125?",
        "options": [
            "A) CWE: $28,170K; CCN: 89050",
            "B) CWE: $62,110K; CCN: 72111",
            "C) CWE: $45,760K; CCN: 14323",
            "D) CWE: $67,130K; CCN: 72111",
            "E) CWE: $45,760K; CCN: 85115"
        ],
        "answer": "C) CWE: $45,760K; CCN: 14323"
    },

    # P1413 - Energy Resilience and Cyber Security of Critical Loads
    {
        "question": "What is the project scope for project P1413 at NAVSTA Rota?",
        "options": [
            "A) Repair and reconfigure unaccompanied housing to comply with CNO directives and eliminate Poor condition ratings",
            "B) Install an Industrial Controls System (ICS) integrating SCADA, DDC, GENSETs, and AMI workstations onto a common communications network",
            "C) Construct a multi-story Bachelor Enlisted Quarters for Permanent Party Personnel E4 and below",
            "D) Repair medium and high voltage electrical distribution system in the Weapons and PWD areas",
            "E) Upgrade air freight terminal hazardous cargo storage by converting unused office space into fire-rated storage rooms"
        ],
        "answer": "B) Install an Industrial Controls System (ICS) integrating SCADA, DDC, GENSETs, and AMI workstations onto a common communications network"
    },
    {
        "question": "What is the Region Mission Alignment score and description for project P1413?",
        "options": [
            "A) Score 4 – Project directly supports CNO guidance specific to Blue Arctic theater through increased mission capabilities",
            "B) Score 3 – Project contributes to closing a critical capability gap identified in an approved Enterprise or Theater planning document",
            "C) Score 3 – Project will impact the three utilities at NAVSTA Rota by integrating stand-alone SCADA control systems into a common fiber optic network",
            "D) Score 4 – Significantly and directly supports COCOM Mission capability requirement when Navy is the executive agent",
            "E) Score 2 – Status quo requires inefficient workarounds; Rota presently has intelligent facility systems that are not interconnected and not maximizing efficiency"
        ],
        "answer": "E) Score 2 – Status quo requires inefficient workarounds; Rota presently has intelligent facility systems that are not interconnected and not maximizing efficiency"
    },

    # P1602 - Expeditionary Camp Unaccompanied Housing
    {
        "question": "What is the impact if not provided for project P1602?",
        "options": [
            "A) Without investment, NAVSTA Rota will continue to operate with an antiquated and unsafe electrical system costly to maintain",
            "B) Unaccompanied sailors will continue to bed down in BQs within a sited ESQD Arc created by adjacent ordnance operations on the Port area",
            "C) Junior enlisted personnel will continue to live in deteriorated housing with major facility deficiencies",
            "D) Navy will not be able to consolidate multiple ICS networks into a single robust communication network",
            "E) Without a right-sized multipurpose logistics facility, NAVEUR's ability to support Task Force objectives will be severely degraded"
        ],
        "answer": "B) Unaccompanied sailors will continue to bed down in BQs within a sited ESQD Arc created by adjacent ordnance operations on the Port area"
    },
    {
        "question": "What is the RAC rating and the Region Severity score/description for project P1602?",
        "options": [
            "A) RAC III; Score 3 – Primary mission accomplished with significant workarounds; 25 emergency trouble calls in the past year",
            "B) RAC II; Score 3 – Serious injury or damage risk to life or environment; BQs in Camp Mitchell are within ESQD ARCs generated by ordnance handling at the Pier",
            "C) RAC I; Score 4 – No workarounds exist to accommodate NAVEUR load plan with existing munition tunnels",
            "D) RAC II; Score 4 – Mission accomplished with significant workarounds; rooms doubled in density, reducing personal SF allowance by half",
            "E) RAC III; Score 2 – Primary mission accomplished with moderate workarounds; status quo requires workaround to maintain capability"
        ],
        "answer": "B) RAC II; Score 3 – Serious injury or damage risk to life or environment; BQs in Camp Mitchell are within ESQD ARCs generated by ordnance handling at the Pier"
    },

    # P1616 - Construct UH BEQ for Perm Party
    {
        "question": "What is the mission of the facility for project P1616?",
        "options": [
            "A) Provides required berthing for Expeditionary Forces stationed at NAVSTA Rota",
            "B) Provide suitable and adequately configured Unaccompanied Housing for Permanent Party Sailors assigned to NAVSTA Rota, driven by the increase of two additional FDNF DDG ships and FOC of HSM79",
            "C) Facility provides housing for 480 junior enlisted personnel (E1-E4) in 240 two-person rooms at NAVSUPPACT Bahrain",
            "D) Provide 24/7 capability supporting Navy Radio Transmitter Facility and global Warfighter communications operations",
            "E) The facility provides operations space for AIMD to conduct maintenance and repair services for aeronautical ground components"
        ],
        "answer": "B) Provide suitable and adequately configured Unaccompanied Housing for Permanent Party Sailors assigned to NAVSTA Rota, driven by the increase of two additional FDNF DDG ships and FOC of HSM79"
    },
    {
        "question": "What is the Region Urgency score and description for project P1616?",
        "options": [
            "A) Score 3 – Resourcing required to meet BY requirement; dysfunctional facility will continue to create operational constraints",
            "B) Score 4 – Failure to execute will result in critical warfighter capability gap; further BEQ degradation will force junior personnel off-base",
            "C) Score 4 – Project requires BY investment in direct support of Permanent Party UH BEQ requirements per the 2023 CNIC-issued R19 for NAVSTA Rota, with a deficit of 260 beds",
            "D) Score 2 – The indoor pool provides water qualification training and a long-term shutdown will impact readiness",
            "E) Score 4 – Project must be funded BY due to accelerated degradation; unplanned outages due to equipment failures will continue"
        ],
        "answer": "C) Score 4 – Project requires BY investment in direct support of Permanent Party UH BEQ requirements per the 2023 CNIC-issued R19 for NAVSTA Rota, with a deficit of 260 beds"
    },

    # RM15-0945 - Repair and Reconfigure Q3 UH Bldg 263
    {
        "question": "What is the project scope for RM15-0945 at NAVSUPPACT Bahrain?",
        "options": [
            "A) Repair potable and fire water lines by replacing underground service pipe, valves, and fittings at NSA-I",
            "B) Repair HVAC, electrical, plumbing, fire protection, electronic entry control locks, doors, fixtures, conveying equipment, and interior finishes; reconfigure 2+2 layout to 2+0",
            "C) Repair building deficiencies for the AIMD complex including major repairs to building envelopes, HVAC, mechanical, and electrical systems",
            "D) Repair and renovate the exterior and interior of munitions tunnel YK-9 and construct a reinforced concrete apron",
            "E) Reconfigure internal walls and repair HVAC, electrical, plumbing, fire protection, entry control locks, doors, fixtures, conveying equipment, and interior finishes of UH B263"
        ],
        "answer": "E) Reconfigure internal walls and repair HVAC, electrical, plumbing, fire protection, entry control locks, doors, fixtures, conveying equipment, and interior finishes of UH B263"
    },
    {
        "question": "What are the Overall Capacity Rating and the facility Condition Rating for RM15-0945?",
        "options": [
            "A) Overall Capacity Rating: 116; Condition Rating: 66",
            "B) Overall Capacity Rating: 100; Condition Rating: 76",
            "C) Overall Capacity Rating: 109; Condition Rating: 80",
            "D) Overall Capacity Rating: 116; Condition Rating: 78",
            "E) Overall Capacity Rating: 88; Condition Rating: 78"
        ],
        "answer": "A) Overall Capacity Rating: 116; Condition Rating: 66"
    },

    # RM15-0946 - Repair Q3 UH Bldg 266
    {
        "question": "What is the mission of the facility for project RM15-0946?",
        "options": [
            "A) Facility provides housing for 480 junior enlisted personnel (E1-E4) in 240 two-person rooms",
            "B) Facility provides housing for 344 junior enlisted personnel (E1-E4) in 172 two-person rooms; Building 266 also houses Naval Security Forces and the Installation EOC",
            "C) The Bachelor Enlisted Quarter 170 provides 219 Bachelor Housing spaces in support of the Base mission",
            "D) Building 623 provides an additional 336 Bachelor Housing spaces in support of the base's mission",
            "E) Facility provides housing for 260 Permanent Party Sailors E4 and below at NAVSTA Rota"
        ],
        "answer": "B) Facility provides housing for 344 junior enlisted personnel (E1-E4) in 172 two-person rooms; Building 266 also houses Naval Security Forces and the Installation EOC"
    },
    {
        "question": "What is the Region Mission Alignment score and the RAC for project RM15-0946?",
        "options": [
            "A) Score 3; RAC II",
            "B) Score 4; RAC III",
            "C) Score 3; RAC III",
            "D) Score 4; RAC I",
            "E) Score 2; RAC III"
        ],
        "answer": "B) Score 4; RAC III"
    },

    # RM16-0798 - Repair Building 2100, Niscemi
    {
        "question": "What is the project scope for RM16-0798 at NAS Sigonella?",
        "options": [
            "A) Repair and expand by 2,766 SF the Air Passenger Terminal located at Building 436 at NAS Sigonella NAS II",
            "B) Repair facility configuration requirements by renovating Building 2100 for Security personnel, including command and control space, armory, communication room, berthing, dog kennels, and self-service galley",
            "C) Repair building deficiencies listed in the ICAP report for the AIMD complex, including major repairs to building envelopes, HVAC, mechanical, and electrical systems",
            "D) Correct Sanitary Survey findings and repair NAS Sigonella's critical water system, including Water Treatment Plants' mechanical and IT infrastructure",
            "E) Repair and increase hazardous cargo storage space by converting unused office and mezzanine space into seven UFC-compliant fire-rated storage rooms"
        ],
        "answer": "B) Repair facility configuration requirements by renovating Building 2100 for Security personnel, including command and control space, armory, communication room, berthing, dog kennels, and self-service galley"
    },
    {
        "question": "What is the Lead Proponent Readiness Support score and description for project RM16-0798?",
        "options": [
            "A) Score 3 – Project will impact the three utilities at NAVSTA Rota by integrating stand-alone SCADA control systems",
            "B) Score 4 – Provides Facilities for ATFP personnel to protect two National Security and DoD missions in Europe for US, DoD, and Allied worldwide communications; MDI 98",
            "C) Score 4 – This project supports Navy AIMD Quality of Service in a forward operating area with no off-base alternative",
            "D) Score 3 – NAS Sigonella Air Terminals supports movement of passengers in support of military force requirements for multiple COCOMs",
            "E) Score 2 – UH project performs critical construction or repairs in support of barracks population in a fair facility"
        ],
        "answer": "B) Score 4 – Provides Facilities for ATFP personnel to protect two National Security and DoD missions in Europe for US, DoD, and Allied worldwide communications; MDI 98"
    },

    # RM16-0799 - Repair B426, B546, B459 & B460 AIMD Hangar & Shops Bldgs
    {
        "question": "What is the impact if not provided for project RM16-0799?",
        "options": [
            "A) Air Cargo-Freight Terminal Personnel cannot safely perform their duties and do not have appropriate space and capacity to properly store mail boxes containing hazardous materials",
            "B) Deterioration and insufficient capacity of primary building systems will worsen conditions in already degraded workspace; more equipment will need to be sent to other bases for repairs, increasing costs",
            "C) Without investment, the Naval Station will continue to operate with an antiquated and unsafe electrical system, costly to maintain and operate",
            "D) NAS Sigonella water systems will continue to operate under a conditional certificate with risk for contamination from raw water backwashing",
            "E) Dysfunctional facility will continue to create operational constraints and inefficient air passenger operations"
        ],
        "answer": "B) Deterioration and insufficient capacity of primary building systems will worsen conditions in already degraded workspace; more equipment will need to be sent to other bases for repairs, increasing costs"
    },
    {
        "question": "What is the Overall Capacity Rating and the COCOM for project RM16-0799?",
        "options": [
            "A) Overall Capacity Rating: 100; COCOM: CENTCOM",
            "B) Overall Capacity Rating: 66; COCOM: EUCOM",
            "C) Overall Capacity Rating: 97; COCOM: EUCOM",
            "D) Overall Capacity Rating: 26; COCOM: EUCOM",
            "E) Overall Capacity Rating: 100; COCOM: EUCOM"
        ],
        "answer": "D) Overall Capacity Rating: 26; COCOM: EUCOM"
    },

    # RM17-0117 - Upgrade Air Freight Terminal Hazardous Cargo Storage B438
    {
        "question": "What is the mission of the facility for project RM17-0117?",
        "options": [
            "A) The Air Passenger Terminal at NAS Sigonella NAS II is open 24 hours a day and provides movement of passengers in support of military force requirements for multiple COCOMs",
            "B) Facility Building 438 is the NAS Sigonella Air Freight Terminal Storage Warehouse, temporarily storing all goods loaded and unloaded by aircraft landing and taking off the Military Airport, for all EURAFSWA Area missions",
            "C) Facility provides operations space for AIMD to conduct maintenance and repair services for aeronautical ground components and equipment in support of P-8 squadrons",
            "D) Provide 24/7 capability supporting Navy Radio Transmitter Facility and global Warfighter communications operations",
            "E) The water treatment plants at NAS I and NAS II produce all potable water for personnel who work and live on the installation"
        ],
        "answer": "B) Facility Building 438 is the NAS Sigonella Air Freight Terminal Storage Warehouse, temporarily storing all goods loaded and unloaded by aircraft landing and taking off the Military Airport, for all EURAFSWA Area missions"
    },
    {
        "question": "What is the CWE and the Region Severity score for project RM17-0117?",
        "options": [
            "A) CWE: $4,014K; Score 3",
            "B) CWE: $11,602K; Score 4",
            "C) CWE: $5,707K; Score 3",
            "D) CWE: $5,828K; Score 3",
            "E) CWE: $5,707K; Score 4"
        ],
        "answer": "C) CWE: $5,707K; Score 3"
    },

    # RM17-1027 - Upgrade Air Passenger Terminal Bldg 436
    {
        "question": "What is the project scope for RM17-1027 at NAS Sigonella?",
        "options": [
            "A) Repair and increase hazardous cargo storage space by converting unused office and mezzanine space into seven UFC-compliant fire-rated storage rooms",
            "B) Repair and expand by 2,766 SF the Air Passenger Terminal located at Building 436 at NAS Sigonella NAS II",
            "C) Repair the AIMD complex building deficiencies including major repairs to building envelopes, HVAC, mechanical, and electrical systems, and construct a prefabricated clean room",
            "D) Repair facility configuration for Security personnel including command and control, armory, communication room, berthing, dog kennels, and self-service galley",
            "E) Repair potable and fire water lines, replacing underground service pipe, valves, and fittings at NSA-I"
        ],
        "answer": "B) Repair and expand by 2,766 SF the Air Passenger Terminal located at Building 436 at NAS Sigonella NAS II"
    },
    {
        "question": "What is the RAC and Region Mission Alignment score for project RM17-1027?",
        "options": [
            "A) RAC: III; Score 3",
            "B) RAC: I; Score 4",
            "C) RAC: II; Score 3",
            "D) RAC: I; Score 3",
            "E) RAC: II; Score 4"
        ],
        "answer": "B) RAC: I; Score 4"
    },

    # RM18-1324 - Repair Potable Water Network System at NSA-I
    {
        "question": "What is the impact if not provided for project RM18-1324?",
        "options": [
            "A) NSA Souda Bay will fail to provide vital sanitary services to the main site and to US vessels at NMPC",
            "B) Failure to implement the repair will adversely impact the reliability of the water and fire distribution within the original section of the base supporting CTF-56, CTF-51/5, and other US/Coalition personnel supporting the 5th fleet operations",
            "C) NAS Sigonella water systems will continue to operate under a conditional certificate with risk for contamination from raw water backwashing",
            "D) Without investment, NAVSTA Rota will continue to operate with an antiquated and unsafe electrical system",
            "E) Air Cargo-Freight Terminal Personnel cannot safely perform their duties and do not have appropriate space to store hazardous materials"
        ],
        "answer": "B) Failure to implement the repair will adversely impact the reliability of the water and fire distribution within the original section of the base supporting CTF-56, CTF-51/5, and other US/Coalition personnel supporting the 5th fleet operations"
    },
    {
        "question": "What is the Region Mission Alignment score and description for project RM18-1324?",
        "options": [
            "A) Score 4 – Significantly and directly supports CENTCOM capability requirements on a daily basis; supports approximately 4,500 PN from various missions",
            "B) Score 3 – There have been ten outages per year due to urgent or emergency repairs; two substantial leaks lost at least 207,250 m3 of potable water",
            "C) Score 4 – Project directly supports forward assigned sailors at a location supporting U.S. and coalition maritime operations at the only MOB in the CENTCOM AOR",
            "D) Score 3 – System shortcomings due to age annually lead to outage with consequent negative impact on NAVSTA Rota mission success",
            "E) Score 2 – Status quo requires inefficient workarounds; facility systems are not interconnected and not maximizing efficiency"
        ],
        "answer": "B) Score 3 – There have been ten outages per year due to urgent or emergency repairs; two substantial leaks lost at least 207,250 m3 of potable water"
    },

    # RM18-2383 - Munitions Tunnel Improvements
    {
        "question": "What is the mission of the facility for project RM18-2383?",
        "options": [
            "A) Wastewater collection and treatment systems that serve both Main Site and NATO Marathi Pier Complex",
            "B) Provides high voltage electrical service for NAVSUP FLC Rota, DLA Bulk Storage, Navy Munitions Command Weapons Storage Compound, and NAVSTA Rota Headquarters",
            "C) Munitions tunnel YK-9 provides adequate storage for the type, size, amount, and classification of ordnance identified in the NAVEUR load plan",
            "D) Facility Building 438 is the NAS Sigonella Air Freight Terminal Storage Warehouse for all EURAFSWA Area missions",
            "E) The facility provides operations space for AIMD to conduct maintenance and repair services for aeronautical ground components"
        ],
        "answer": "C) Munitions tunnel YK-9 provides adequate storage for the type, size, amount, and classification of ordnance identified in the NAVEUR load plan"
    },
    {
        "question": "What is the CWE and Installation for project RM18-2383?",
        "options": [
            "A) CWE: $5,828K; NAVSUPPACT Bahrain",
            "B) CWE: $4,453K; NAVSUPPACT Souda Bay",
            "C) CWE: $1,898K; NAVSUPPACT Souda Bay",
            "D) CWE: $1,898K; NAS Sigonella",
            "E) CWE: $7,147K; NAVSTA Rota"
        ],
        "answer": "C) CWE: $1,898K; NAVSUPPACT Souda Bay"
    },

    # RM19-0930 - HV System Repairs, Feeders #2, 3, 4, and 9
    {
        "question": "What is the project scope for RM19-0930 at NAVSTA Rota?",
        "options": [
            "A) Repair the medium and high voltage electrical distribution system in the Airfield and Industrial Facilities area, including undergrounding of 15kV feeders and installation of new SF6 pad-mounted switchgears",
            "B) Repair the medium and high voltage electrical distribution system in the Weapons, PWD, Retail, Core, and Fuel areas; install 278 power poles and associated copper conductors",
            "C) Upgrade and repair non-compliant aspects of the Jerez entry control point to meet Unified Facilities Criteria",
            "D) Repair the medium and high voltage distribution system supplying the Airfield and Air Operations facilities and EOD facilities",
            "E) Repair and renovate the wastewater collection and treatment systems at multiple locations at NSA Souda Bay Main Site and NMPC"
        ],
        "answer": "B) Repair the medium and high voltage electrical distribution system in the Weapons, PWD, Retail, Core, and Fuel areas; install 278 power poles and associated copper conductors"
    },
    {
        "question": "What is the Region Severity score/description and the Project CCN for RM19-0930?",
        "options": [
            "A) Score 4 – System shortcomings due to age annually lead to outage; Feeders identified support critical Base infrastructure areas (airfield, Weapons Compound); CCN: 81232",
            "B) Score 3 – System shortcomings due to age annually lead to outage; Feeders identified support critical Base infrastructure areas (airfield, Weapons Compound); CCN: 81231",
            "C) Score 4 – Significant arc flash can occur during operation of medium voltage switches; ocean winds create recurring outages due to salt and sand deposits; CCN: 13620",
            "D) Score 3 – System shortcomings due to age annually lead to outage; feeders identified support critical Base infrastructure; CCN: 81231",
            "E) Score 4 – Antiquated distribution system is past its designed life expectancy; considerable lengths along the waterfront leading to increased environmental degradation; CCN: 81231"
        ],
        "answer": "D) Score 3 – System shortcomings due to age annually lead to outage; feeders identified support critical Base infrastructure; CCN: 81231"
    },

    # RM20-0418 - NAS II Water System Redundancy & Resiliency Repairs
    {
        "question": "What is the mission of the facility for project RM20-0418?",
        "options": [
            "A) Repair potable and fire water lines serving NSA I facilities",
            "B) The water treatment plants at NAS I and NAS II produce all potable water for personnel who work and live on the installation, and provide potable water for critical facilities such as the hospital, airfield, and flightline clinic",
            "C) Wastewater collection and treatment systems that serve both Main Site and NATO Marathi Pier Complex",
            "D) Facility provides housing for junior enlisted personnel at NAS Sigonella",
            "E) Provide 24/7 capability supporting Navy Radio Transmitter Facility and global Warfighter communications operations"
        ],
        "answer": "B) The water treatment plants at NAS I and NAS II produce all potable water for personnel who work and live on the installation, and provide potable water for critical facilities such as the hospital, airfield, and flightline clinic"
    },
    {
        "question": "What is the Region Readiness Support score and description for project RM20-0418?",
        "options": [
            "A) Score 3 – Project directly supports forward assigned sailors at a location that supports U.S. and coalition maritime operations at the only MOB in the CENTCOM AOR",
            "B) Score 4 – NSA Sigonella has no access to public drinking water; project ensures automatic backwash of treated water prevents biological growth, mitigating contamination associated with existing manual backwash operations",
            "C) Score 4 – Improves ability/resilience to rearm/resupply in forward area; Souda Bay is at 50% capacity requiring two magazines to pre-position and store long ordnance",
            "D) Score 3 – Project directly supports forward assigned sailors and significantly and directly supports CNO policy to house all single shipboard sailors E1-E4 on base",
            "E) Score 4 – Project directly supports CNO guidance; backflow prevention program was a risk finding at Red Hill; NSA Sigonella is 1 of 4 locations in EURAFCENT that relies on own well water with no backup"
        ],
        "answer": "E) Score 4 – Project directly supports CNO guidance; backflow prevention program was a risk finding at Red Hill; NSA Sigonella is 1 of 4 locations in EURAFCENT that relies on own well water with no backup"
    },

    # RM20-0429 - Repair Unaccompanied Housing (UH) Bldg 623
    {
        "question": "What is the impact if not provided for project RM20-0429 at NAS Sigonella?",
        "options": [
            "A) Air Cargo-Freight Terminal Personnel cannot safely perform their duties and lack appropriate space to store hazardous mail boxes",
            "B) The dysfunctional facility will continue to create operational constraints and inefficient air passenger operations",
            "C) Failure to perform this work will maintain current inadequate conditions; continued deterioration could become a health hazard; without investment, junior enlisted personnel will continue to live in deteriorated housing and could be forced off-base at approximately $8.00M/year",
            "D) Without this project, junior enlisted personnel will continue to live in deteriorated housing that could force the installation to rent off-base space at approximately $5.22M/year",
            "E) Without investment, NAVSTA Rota will continue to operate with an antiquated and unsafe electrical system"
        ],
        "answer": "C) Failure to perform this work will maintain current inadequate conditions; continued deterioration could become a health hazard; without investment, junior enlisted personnel will continue to live in deteriorated housing and could be forced off-base at approximately $8.00M/year"
    },
    {
        "question": "What is the Overall Capacity Rating and the facility MDI for project RM20-0429?",
        "options": [
            "A) Overall Capacity Rating: 116; MDI: 78",
            "B) Overall Capacity Rating: 88; MDI: 64",
            "C) Overall Capacity Rating: 100; MDI: 95",
            "D) Overall Capacity Rating: 158; MDI: 64",
            "E) Overall Capacity Rating: 158; MDI: 78"
        ],
        "answer": "D) Overall Capacity Rating: 158; MDI: 64"
    },

    # RM20-0438 - Repair Unaccompanied Housing (UH) Bldg 170
    {
        "question": "What is the project scope for RM20-0438 at NAS Sigonella?",
        "options": [
            "A) Repair the plumbing, HVAC, and electrical systems and provide civil and architectural repair works to the interiors and exteriors of Building 170",
            "B) Reconfigure internal walls and repair HVAC, electrical, plumbing, fire protection, electronic entry control locks, and interior finishes of UH Building 263",
            "C) Repair and expand by 2,766 SF the Air Passenger Terminal located at Building 436",
            "D) Repair and reconfigure UH Building 623 to comply with CNO directives and eliminate Poor condition",
            "E) Repair HVAC, electrical, plumbing, fire protection, electronic entry control locks, doors, windows, fixtures, conveying equipment, and interior and exterior finishes in BEQ Bldg 266"
        ],
        "answer": "A) Repair the plumbing, HVAC, and electrical systems and provide civil and architectural repair works to the interiors and exteriors of Building 170"
    },
    {
        "question": "What is the CWE and facility Condition Rating for project RM20-0438?",
        "options": [
            "A) CWE: $6,998K; Condition Rating: 84",
            "B) CWE: $4,684K; Condition Rating: 78",
            "C) CWE: $4,684K; Condition Rating: 80",
            "D) CWE: $11,470K; Condition Rating: 73",
            "E) CWE: $18,954K; Condition Rating: 66"
        ],
        "answer": "C) CWE: $4,684K; Condition Rating: 80"
    },

    # RM20-0534 - Bonifaz St Electrical Distribution Undergrounding
    {
        "question": "What is the mission of the facility for project RM20-0534?",
        "options": [
            "A) Provides high voltage electrical service for NAVSUP FLC Rota, DLA Bulk Storage, Navy Munitions Command Weapons Storage Compound, NAVSTA Rota Headquarters, and Core Operational Facilities",
            "B) The project will enhance system reliability and efficiency with a correctly sized and greatly stabilized electrical distribution system",
            "C) Provide for the safe check of personnel and vehicle identification entering and exiting the installation and quickly responding to and containing threats",
            "D) The indoor and outdoor pool provides recreation, fitness opportunities, youth sports activities, and water qualification training to operational units at NAVSTA Rota",
            "E) Wastewater collection and treatment systems that serve both Main Site and NATO Marathi Pier Complex"
        ],
        "answer": "B) The project will enhance system reliability and efficiency with a correctly sized and greatly stabilized electrical distribution system"
    },
    {
        "question": "What is the CWE and the Region Mission Alignment score for project RM20-0534?",
        "options": [
            "A) CWE: $7,147K; Score 4",
            "B) CWE: $1,998K; Score 3",
            "C) CWE: $3,188K; Score 3",
            "D) CWE: $1,998K; Score 4",
            "E) CWE: $2,981K; Score 2"
        ],
        "answer": "D) CWE: $1,998K; Score 4"
    },

    # RM20-0669 - Wastewater Rehabilitation at Souda Bay Marathi & Main Site
    {
        "question": "What is the project scope for RM20-0669 at NAVSUPPACT Souda Bay?",
        "options": [
            "A) Repair potable and fire water lines serving NSA I facilities, replacing underground service pipe, valves, and fittings",
            "B) Repair and renovate wastewater collection and treatment systems at multiple locations at NSA Souda Bay Main Site and NMPC, including sewer collection system, leach field, STP, and storage tanks",
            "C) Repair and renovate exterior and interior of munitions tunnel YK-9 and construct a reinforced concrete apron",
            "D) Repair and expand by 2,766 SF the Air Passenger Terminal at NAS Sigonella NAS II",
            "E) Corrects Sanitary Survey findings and repairs NAS Sigonella's critical water system, including Water Treatment Plants' mechanical and IT infrastructure"
        ],
        "answer": "B) Repair and renovate wastewater collection and treatment systems at multiple locations at NSA Souda Bay Main Site and NMPC, including sewer collection system, leach field, STP, and storage tanks"
    },
    {
        "question": "What is the Region Severity score/description for project RM20-0669?",
        "options": [
            "A) Score 3 – There have been ten outages per year due to urgent or emergency repairs; two substantial leaks lost at least 207,250 m3 of potable water",
            "B) Score 4 – NSA has experienced 4 sanitary sewer overflows over 5 years; wastewater system failure would require extreme workarounds; portable toilet rentals and 5-10 pump trucks daily at $400/Kg would be required",
            "C) Score 3 – System shortcomings due to age annually lead to outage with negative impact on NAVSTA Rota mission success",
            "D) Score 4 – Delaying this project increases risk of sanitary sewer overflows; structural assessment indicates approximately 2 years of service life left for the west tank at Marathi Piers",
            "E) Score 4 – NSA Souda Bay is the only port able to receive a carrier and treat its sewage in the region"
        ],
        "answer": "D) Score 4 – Delaying this project increases risk of sanitary sewer overflows; structural assessment indicates approximately 2 years of service life left for the west tank at Marathi Piers"
    },

    # RM20-0827 - MWR Pool Repairs and Upgrades
    {
        "question": "What is the impact if not provided for project RM20-0827?",
        "options": [
            "A) NAVSTA Rota will not be in full compliance with Unified Facilities Criteria for entry control points and has to accept the potential risk of increased casualties",
            "B) Operational readiness, youth sports activities, and recreation/QOL provided by the indoor pool will be impacted if the capability needs to be shutdown due to structural damage; project upgrades the outdoor pool to accommodate missions currently supported by the indoor pool",
            "C) Without investment, the Naval Station will continue to operate with an antiquated and unsafe electrical system, costly to maintain and operate",
            "D) The dysfunctional facility will continue to create operational constraints and inefficient air passenger operations at NAS Sigonella",
            "E) Junior enlisted personnel will continue to live in deteriorated housing and could be forced off-base against CNO policy"
        ],
        "answer": "B) Operational readiness, youth sports activities, and recreation/QOL provided by the indoor pool will be impacted if the capability needs to be shutdown due to structural damage; project upgrades the outdoor pool to accommodate missions currently supported by the indoor pool"
    },
    {
        "question": "What is the Project CCN and the Region Readiness Support score for project RM20-0827?",
        "options": [
            "A) CCN: 73025; Score 4",
            "B) CCN: 75030; Score 2",
            "C) CCN: 75030; Score 1",
            "D) CCN: 74053; Score 2",
            "E) CCN: 75030; Score 3"
        ],
        "answer": "C) CCN: 75030; Score 1"
    },

    # RM21-0367 - Jerez Gate ECP Compliance Upgrades
    {
        "question": "What is the mission of the facility for project RM21-0367?",
        "options": [
            "A) The indoor and outdoor pool provides recreation, fitness opportunities, youth sports activities, and water qualification training to operational units at NAVSTA Rota",
            "B) The project will enhance system reliability and efficiency with a correctly sized and greatly stabilized electrical distribution system",
            "C) Provide for the safe check of personnel and vehicle identification entering and exiting the installation and quickly responding to and containing threats",
            "D) Wastewater collection and treatment systems that serve both Main Site and NATO Marathi Pier Complex at NSA Souda Bay",
            "E) Provides high voltage electrical service for NAVSUP FLC Rota, DLA Bulk Storage, and Navy Munitions Command Weapons Storage Compound"
        ],
        "answer": "C) Provide for the safe check of personnel and vehicle identification entering and exiting the installation and quickly responding to and containing threats"
    },
    {
        "question": "What is the Overall Capacity Rating and Region Severity score for project RM21-0367?",
        "options": [
            "A) Overall Capacity Rating: 100; Score 3",
            "B) Overall Capacity Rating: 66; Score 4",
            "C) Overall Capacity Rating: 45; Score 4",
            "D) Overall Capacity Rating: 45; Score 3",
            "E) Overall Capacity Rating: 100; Score 4"
        ],
        "answer": "C) Overall Capacity Rating: 45; Score 4"
    },
    # RM21-0395 - LIFT STATION REHABILITATION - NAVSTA ROTA SP
    {
        "question": "What is the mission of the facility for project RM21-0395?",
        "options": [
            "A) Provide fire suppression systems for aircraft hangars at NAVSTA Rota",
            "B) Proper conveyance of wastewater is critical to support daily operations of NAVSTA Rota; if corroded and rusted pumps are not replaced, catastrophic failure is possible leading to wet well surcharging and sewer backups",
            "C) Provide potable water supply and fire protection suppression capabilities to the piers at NAVSTA Rota",
            "D) Operate a central power plant for the entirety of NAVSTA Rota Base on a 24/7 basis",
            "E) Provide laboratory, administration, and classroom spaces for Navy Environmental Preventative Medical Unit"
        ],
        "answer": "B) Proper conveyance of wastewater is critical to support daily operations of NAVSTA Rota; if corroded and rusted pumps are not replaced, catastrophic failure is possible leading to wet well surcharging and sewer backups"
    },
    {
        "question": "What is the CWE and Project CCN for RM21-0395?",
        "options": [
            "A) CWE $9,218K, CCN 21107",
            "B) CWE $5,667K, CCN 85110",
            "C) CWE $4,916K, CCN 83230",
            "D) CWE $23,876K, CCN 81109",
            "E) CWE $4,215K, CCN 53050"
        ],
        "answer": "C) CWE $4,916K, CCN 83230"
    },

    # RM21-0686 - CORRECT DEFICIENCIES, AV UNIT HANGAR B460, NSA-III - NAVSUPPACT BAHRAIN
    {
        "question": "What is the impact if not provided for project RM21-0686?",
        "options": [
            "A) Sewer system overflows in and around lift stations or sewer backups into buildings and residences",
            "B) Hangar 460 will remain unprotected from possible fire incidents; aircraft will be severely damaged and extensive damage will be caused to personnel, the hangar structure and other connected buildings",
            "C) NSA will be unable to utilize the Intermodal Road for logistical support, raising serious life safety and security concerns",
            "D) The four existing hangars will not have adequate fire suppression systems, placing hundreds of millions of dollars of facilities and equipment at risk",
            "E) NEPMU7 will continue to operate inefficiently in three separate facilities that do not meet their full BFR"
        ],
        "answer": "B) Hangar 460 will remain unprotected from possible fire incidents; aircraft will be severely damaged and extensive damage will be caused to personnel, the hangar structure and other connected buildings"
    },
    {
        "question": "What is the RAC, COCOM, and Lead Proponent for project RM21-0686?",
        "options": [
            "A) RAC II, EUCOM, NAVFAC",
            "B) RAC I, CENTCOM, CNIC N3",
            "C) RAC I, EUCOM, NAVFAC",
            "D) RAC II, CENTCOM, NMC",
            "E) RAC I, TRANSCOM, CNIC N3"
        ],
        "answer": "B) RAC I, CENTCOM, CNIC N3"
    },

    # RM21-0698 - INTERMODAL ACCESS ROAD UPGRADES - NAVSUPPACT SOUDA BAY GR
    {
        "question": "What is the mission of the facility for project RM21-0698?",
        "options": [
            "A) Provide aircraft parking apron for loading, unloading and servicing of aircraft",
            "B) Provide fire water supply for aircraft hangars at Souda Bay",
            "C) Intermodal Road provides the connection from NATO Marathi Pier Complex to NSA Souda Bay for airfield and Main Base logistical support along with access to the bulk fuels JP5 pipeline and communications lines",
            "D) Provide berthing accommodations for tenant commands and assigned personnel",
            "E) Support U.S. port operations and storage of fuel, ammunition, and supplies for the Sixth Fleet"
        ],
        "answer": "C) Intermodal Road provides the connection from NATO Marathi Pier Complex to NSA Souda Bay for airfield and Main Base logistical support along with access to the bulk fuels JP5 pipeline and communications lines"
    },
    {
        "question": "What is the Region Severity score and description for project RM21-0698?",
        "options": [
            "A) Score 3 – Primary mission accomplished with significant workarounds; motors and valves unable to efficiently collect and transfer wastewater",
            "B) Score 2 – The condition of the pumps and valves are currently failing and may include surcharging of lift station wet wells",
            "C) Score 4 – Intermodal Road is critical for munitions, fuel and cargo functions; alternate paths expose local citizens and valuable USN assets to high risk; severe risk due to potential mission failure and catastrophic environmental contamination should road collapse damage JP-5 pipeline",
            "D) Score 4 – Fire could go unnoticed with lack of functional fire suppression; severe threat to life and asset",
            "E) Score 3 – Primary mission accomplished with significant workarounds; no real property space available for housing DDG ship duty sections"
        ],
        "answer": "C) Score 4 – Intermodal Road is critical for munitions, fuel and cargo functions; alternate paths expose local citizens and valuable USN assets to high risk; severe risk due to potential mission failure and catastrophic environmental contamination should road collapse damage JP-5 pipeline"
    },

    # RM21-0774 - MODERNIZE NAS II HANGAR FIRE WATER SYSTEM - NAS SIGONELLA IT
    {
        "question": "What is the project scope for RM21-0774?",
        "options": [
            "A) Repair about 63,300 SF area of Stand 55 including demolition of existing and reconstruction of new concrete pavement",
            "B) Repair and modernization of the existing high-voltage infrastructure system to improve reliability and resiliency of utility and distribution electrical power systems at NAS II",
            "C) Repair the existing inefficient flightline hangars fire water system so that it meets current code by providing a dedicated FW loop and updating the FW pump house",
            "D) Install fire suppression system and perform interior repair works in aircraft maintenance hangar Building 460",
            "E) Reconstruct aircraft parking apron with 13 inches of Portland Cement Concrete over six inches of base"
        ],
        "answer": "C) Repair the existing inefficient flightline hangars fire water system so that it meets current code by providing a dedicated FW loop and updating the FW pump house"
    },
    {
        "question": "What is the Region Operational Cost score and description for project RM21-0774?",
        "options": [
            "A) Score 1 – Business case analysis shows overall savings with a payback of 10-15 years",
            "B) Score 0 – No cost savings have been documented",
            "C) Score 2 – Significant deterioration of equipment impacting reliability; overhead wiring not properly sized in several locations",
            "D) Score 3 – The current FW system cannot suppress an aircraft fire; a fire watch is in place 24/7 costing over $1.5M by end of FY21; over 270K gallons of water lost annually from leaking FW with a payback of 7.6 years",
            "E) Score 4 – Documented detailed cost analysis shows operational cost savings when scheduled overhauls are performed"
        ],
        "answer": "D) Score 3 – The current FW system cannot suppress an aircraft fire; a fire watch is in place 24/7 costing over $1.5M by end of FY21; over 270K gallons of water lost annually from leaking FW with a payback of 7.6 years"
    },

    # RM21-2038 - RENOVATE POWER PLANT UTILITY BUILDING 64 - NAVSTA ROTA SP
    {
        "question": "What is the project scope for RM21-2038?",
        "options": [
            "A) Overhaul prime power engine-generator sets including first and second top-end overhauls and replacement of select generator sets",
            "B) Repair by replacement of the entire Building 64 envelope and reinforcement of the concrete roof to comply with Seismic and ATFP regulations, providing a safe building with a reliable source of power",
            "C) Replace backflow preventers for ship-to-shore water connections and provide potable water storage for pier fire protection",
            "D) Provide general repair to buildings in the Cold Storage Shed and Waterfront Transit Warehouse facilities",
            "E) Convert Building 555 into adequate area for essential ship services including crew computer access, fire watch sleeping areas, heads, and showers"
        ],
        "answer": "B) Repair by replacement of the entire Building 64 envelope and reinforcement of the concrete roof to comply with Seismic and ATFP regulations, providing a safe building with a reliable source of power"
    },
    {
        "question": "What is the condition rating, MDI, and year built for the Power Plant Facility (FAC#64) associated with project RM21-2038?",
        "options": [
            "A) Condition Rating 80, MDI 98, Year Built 1963",
            "B) Condition Rating 73, MDI 84, Year Built 1959",
            "C) Condition Rating 61, MDI 53, Year Built 1958",
            "D) Condition Rating 79, MDI 73, Year Built 1974",
            "E) Condition Rating 44, MDI 14, Year Built 1970"
        ],
        "answer": "B) Condition Rating 73, MDI 84, Year Built 1959"
    },

    # RM22-0299 - BUILDING 8 RENOVATION FOR NEPMU7 MEDICAL LAB - NAVSTA ROTA SP
    {
        "question": "What is the mission of the facility for project RM22-0299?",
        "options": [
            "A) Provide fire suppression systems and interior repairs for aircraft maintenance hangars",
            "B) NEPMU-7 supports 3 COCOMs (AFRICOM, EUCOM, and CENTCOM) with public health surveillance, providing preventive medicine, environmental health, entomology, industrial hygiene and surveillance laboratory services",
            "C) Provide crew berthing accommodations that meet UFC building standards including fire protection systems",
            "D) Operate the wastewater collection and treatment systems serving the entire Naval Air Station",
            "E) Support intermodal operations including fuel and ordnance transportation between the pier complex and main base"
        ],
        "answer": "B) NEPMU-7 supports 3 COCOMs (AFRICOM, EUCOM, and CENTCOM) with public health surveillance, providing preventive medicine, environmental health, entomology, industrial hygiene and surveillance laboratory services"
    },
    {
        "question": "What is the overall capacity rating and current space utilization situation for project RM22-0299?",
        "options": [
            "A) Capacity rating 81; facilities averaging condition rating of 72 with one building in failing (58) condition",
            "B) Capacity rating 77; installation runs at approximately 90% occupancy rate with a 432,598 SF deficit of adequate berthing",
            "C) Capacity rating 56; NEPMU-7 BFR justifies 12,219 SF but current utilization across multiple separate facilities is only 7,110 SF",
            "D) Capacity rating 39; facility has 33,578 SF with condition rating of 76",
            "E) Capacity rating 100; single 12-inch main currently feeds all piers with potable water"
        ],
        "answer": "C) Capacity rating 56; NEPMU-7 BFR justifies 12,219 SF but current utilization across multiple separate facilities is only 7,110 SF"
    },

    # RM22-0530 - REPAIR STAND 55 IN NSA-III BAHRAIN - NAVSUPPACT BAHRAIN
    {
        "question": "What is the mission of the facility for project RM22-0530?",
        "options": [
            "A) Provide aircraft intermediate maintenance support for helicopter and fixed-wing aircraft squadrons",
            "B) Aircraft Parking Apron required for loading, unloading and servicing of aircraft, and providing parking space to support mission and flight operations for transfer of expeditionary and peacetime personnel throughout CENTCOM's AOR",
            "C) Provide ordnance support to NAVEUR, AFRICOM, USFF Command, and DOD conventional ammunition requirements",
            "D) Support U.S. port operations and storage of fuel, ammunition, and supplies for the Sixth Fleet and transiting ships",
            "E) Provide fire protection for aircraft hangars and ensure compliance with UFC-3-600-01"
        ],
        "answer": "B) Aircraft Parking Apron required for loading, unloading and servicing of aircraft, and providing parking space to support mission and flight operations for transfer of expeditionary and peacetime personnel throughout CENTCOM's AOR"
    },
    {
        "question": "What is the Region Severity score and description for project RM22-0530?",
        "options": [
            "A) Score 3 – Primary mission accomplished with significant workarounds; certain areas of the facility are off limits or avoided due to damage and deterioration",
            "B) Score 4 – The pavement condition of Stand 55 will no longer support wide body aircraft parking; SME PCI survey in FY2017 determined the necessary level of service did not meet standard requirements",
            "C) Score 4 – West End CLUs have reached end of life expectancy with multiple building system issues and a RAC 1 rating due to lack of fire suppression",
            "D) Score 2 – The condition of pumps, valves, and piping are currently failing and will fail in the near future",
            "E) Score 4 – Apron already demonstrates very poor pavement condition with PCI of 25; repairs vital to ensure continued operations"
        ],
        "answer": "B) Score 4 – The pavement condition of Stand 55 will no longer support wide body aircraft parking; SME PCI survey in FY2017 determined the necessary level of service did not meet standard requirements"
    },

    # RM23-0370 - REPAIR WEST END CLUS - CAMP LEMONNIER DJIBOUTI
    {
        "question": "What is the project scope for RM23-0370?",
        "options": [
            "A) Provide general repair to buildings in the Cold Storage Shed and Waterfront Transit Warehouse facilities at Augusta Bay",
            "B) Demolish 153 single occupancy and 21 double occupancy wet CLUs in the West End CLU block and construct 162 single occupancy and 24 double occupancy wet CLUs within the same footprint",
            "C) Overhaul prime power engine-generator sets 5, 6, 7, and 8 and replace engine-generator sets 13, 15, and 18",
            "D) Convert Building 555 into adequate area for essential ship services including sleeping areas, heads, showers, and crew computer access",
            "E) Repair and modernize existing high-voltage infrastructure to improve reliability and resiliency of electrical power systems"
        ],
        "answer": "B) Demolish 153 single occupancy and 21 double occupancy wet CLUs in the West End CLU block and construct 162 single occupancy and 24 double occupancy wet CLUs within the same footprint"
    },
    {
        "question": "What deficiency code and severity is documented for the West End CLU facilities in project RM23-0370?",
        "options": [
            "A) W2 Code Compliance – Seismic, Severity 2: Vulnerability study says building will likely fail under design earthquake load",
            "B) X3 Functional or Space Criteria – Building or Structure, Severity 2: Pavement condition does not meet necessary level of service",
            "C) W5 Code Compliance – Fire Codes, Severity 4: RAC 1 rating assessed for lack of fire suppression system and fire detection system in berthing",
            "D) W7 Code Compliance – ATFP, Severity 2: Parking next to building does not meet standoff requirements",
            "E) Z9 Inadequate Capacity/Coverage – Supporting Systems, Severity 2: Systems cannot meet current daily sanitary sewage output"
        ],
        "answer": "C) W5 Code Compliance – Fire Codes, Severity 4: RAC 1 rating assessed for lack of fire suppression system and fire detection system in berthing"
    },

    # RM23-0509 - REPAIR LOGISTICS SUPPORT FACILITIES - AUGUSTA BAY - NAS SIGONELLA IT
    {
        "question": "What is the mission of the facility for project RM23-0509?",
        "options": [
            "A) Provide 24/7 mission-critical communication requirements for shore to ship High Frequency and Very Low Frequency communications for U.S., NATO, and coalition forces",
            "B) Augusta Bay supports U.S. port operations and storage of fuel, ammunition, and supplies in support of the Sixth Fleet and transiting U.S. ships",
            "C) Provide aircraft parking apron to support transfer of expeditionary and peacetime personnel throughout CENTCOM's AOR",
            "D) Process sanitary sewage and provide wastewater collection and treatment for the entire Naval Air Station Sigonella",
            "E) Provide ordnance support to NAVEUR, AFRICOM, and USFF Command within the area of responsibility"
        ],
        "answer": "B) Augusta Bay supports U.S. port operations and storage of fuel, ammunition, and supplies in support of the Sixth Fleet and transiting U.S. ships"
    },
    {
        "question": "What is the Region Mission Alignment score and description for project RM23-0509?",
        "options": [
            "A) Score 2 – This project addresses a major infrastructure capability gap identified in the 2020 NAS Sigonella Region Master Plan",
            "B) Score 4 – This is a critical mission alignment requirement that strengthens the resilience of the high voltage infrastructure identified by the PWD UEM",
            "C) Score 3 – Project contributes to mitigating identified COCOM capability gap to respond to OPLAN and contingency events as identified in the Regional Master Plan and the Installation IDP",
            "D) Score 4 – Supports sustainment and long-term strategic joint force lethality; aligns with the National Defense Strategy prioritization of prepositioned forward stocks",
            "E) Score 3 – Project contributes to CNO directed basing decision and mission growth of homeported FNDF DDGs at NAVSTA Rota"
        ],
        "answer": "C) Score 3 – Project contributes to mitigating identified COCOM capability gap to respond to OPLAN and contingency events as identified in the Regional Master Plan and the Installation IDP"
    },

    # RM23-0513 - HIGH VOLTAGE ELECTRICAL SYSTEM RECAPITALIZATION (NAS II) - NAS SIGONELLA IT
    {
        "question": "What is the impact if not provided for project RM23-0513?",
        "options": [
            "A) ECMs will need to be kept vacant and security fencing that does not meet EUCOM force protection criteria increases installation vulnerability",
            "B) The obsolete high-voltage system will continue to deteriorate; deficiencies will lead to major failures and significant risk to mission; the installation will not be able to meet the required level 2 operational capability",
            "C) Untreated wastewater discharged into the environment will cause pollution potentially leading to environmental violations and negative host nation relations",
            "D) Critical airfield pavement will continue to create FOD concerns and degrade at an accelerated rate",
            "E) NAVSTA Rota will lose power at Building 64 which represents a single point of failure for the entire installation"
        ],
        "answer": "B) The obsolete high-voltage system will continue to deteriorate; deficiencies will lead to major failures and significant risk to mission; the installation will not be able to meet the required level 2 operational capability"
    },
    {
        "question": "What are the facility condition rating, MDI, and PRV for the Electrical Distribution asset (24KVUG) in project RM23-0513?",
        "options": [
            "A) Condition Rating 43, MDI 92, PRV $667.8K",
            "B) Condition Rating 100, MDI 84, PRV $9,928.2K",
            "C) Condition Rating 69, MDI 84, PRV $9,928.2K",
            "D) Condition Rating 80, MDI 72, PRV $597.0K",
            "E) Condition Rating 79, MDI 73, PRV $2,381.3K"
        ],
        "answer": "C) Condition Rating 69, MDI 84, PRV $9,928.2K"
    },

    # RM23-0514 - WEAPONS COMPOUND IMPROVEMENTS - NAS SIGONELLA IT
    {
        "question": "What is the mission of the facility for project RM23-0514?",
        "options": [
            "A) Provide preventive medicine, environmental health, and surveillance laboratory services to support mission and force health protection",
            "B) Provide aircraft intermediate maintenance support for helicopters and fixed-wing aircraft",
            "C) Naval Munitions Command Atlantic Detachment Sigonella provides ordnance support to NAVEUR, AFRICOM, USFF Command, and DOD conventional ammunition requirements within the AOR, and provides quick response team support for VLS/CLS operations",
            "D) Provide 24/7 mission-critical communication requirements for High Frequency, Low Frequency, and Very Low Frequency communications for U.S., NATO, and coalition forces",
            "E) Provide electrical power via expeditionary diesel engine-generator sets for 90 percent of base power requirements"
        ],
        "answer": "C) Naval Munitions Command Atlantic Detachment Sigonella provides ordnance support to NAVEUR, AFRICOM, USFF Command, and DOD conventional ammunition requirements within the AOR, and provides quick response team support for VLS/CLS operations"
    },
    {
        "question": "What is the Region Readiness Support score and description for project RM23-0514?",
        "options": [
            "A) Score 4 – No alternate source of power for NAS Sigonella NAS II; unplanned outage would incur more than 8 hours to repair",
            "B) Score 3 – Average condition rating of the three storage facilities is 72 with one building in failing condition; primary mission accomplished with moderate workarounds",
            "C) Score 4 – ECMs are in failing condition and only permitted to store up to 50% of originally designed capacity; NAS Sigonella serves as a key ordnance storage location and designated secondary ordnance load out site; mission readiness remains degraded without RM23-0514",
            "D) Score 4 – CLDJ has no back-up power, increasing risk of power black-outs affecting OPLAN and contingency mission capabilities",
            "E) Score 3 – Fully meets BFR with current capacity of 70% or less; NEPMU-7 current utilization is only 58% of requirements"
        ],
        "answer": "C) Score 4 – ECMs are in failing condition and only permitted to store up to 50% of originally designed capacity; NAS Sigonella serves as a key ordnance storage location and designated secondary ordnance load out site; mission readiness remains degraded without RM23-0514"
    },

    # RM23-0634 - CONVERT BLDG 555 FOR FNDF SHIPS AVAIL REQUIREMENTS - NAVSTA ROTA SP
    {
        "question": "What is the mission of the facility for project RM23-0634?",
        "options": [
            "A) Provide fire protection for aircraft operating in hangars at NAVSTA Rota in compliance with UFC-3-600-01",
            "B) Provide essential crew member operational space and sleeping area in support of DESRON 60 DDG maintenance operations, directly supporting Destroyer Squadron SIX ZERO Fleet Technical Assistance and intrusive maintenance of DDGs homeported at NAVSTA Rota",
            "C) Provide wastewater collection and treatment services for the entire installation",
            "D) Provide laboratory, administration, classroom, and conference room spaces for NEPMU7",
            "E) Operate as central power plant for NAVSTA Rota Base on a 24/7 basis"
        ],
        "answer": "B) Provide essential crew member operational space and sleeping area in support of DESRON 60 DDG maintenance operations, directly supporting Destroyer Squadron SIX ZERO Fleet Technical Assistance and intrusive maintenance of DDGs homeported at NAVSTA Rota"
    },
    {
        "question": "What is the estimated annual cost savings and payback period documented for project RM23-0634?",
        "options": [
            "A) Annual savings of $900K; project provides reject water source that can be mixed and used for irrigation",
            "B) Annual savings of $1.5M from fire watch elimination; payback period of 7.6 years",
            "C) Annual cost of approximately $1M for trailers; with project CWE of $4,696K, estimated payback is approximately 4.696 years",
            "D) Annual savings realized from repair costs; repair costs increase without maintenance as deeper deterioration leads to catastrophic failure",
            "E) No documented cost savings; project will not save operational cost but will eliminate life safety risk"
        ],
        "answer": "C) Annual cost of approximately $1M for trailers; with project CWE of $4,696K, estimated payback is approximately 4.696 years"
    },

    # RM23-0697 - WASTEWATER TREATMENT PLANT UPGRADES (NAS II) - NAS SIGONELLA IT
    {
        "question": "What is the impact if not provided for project RM23-0697?",
        "options": [
            "A) Lift stations will experience wet well surcharging, sewer system overflows, or sewer backups into buildings",
            "B) Work outages will continue and untreated wastewater discharged into the environment will cause pollution potentially leading to environmental violations and negative host nation relations",
            "C) ECMs will need to be kept vacant and security fencing will not meet EUCOM force protection criteria",
            "D) The four existing hangars will not have adequate fire suppression systems, placing hundreds of millions of dollars of facilities and equipment at risk",
            "E) Generator failures will require a 6-12 month lead time to resolve, affecting base and mission execution"
        ],
        "answer": "B) Work outages will continue and untreated wastewater discharged into the environment will cause pollution potentially leading to environmental violations and negative host nation relations"
    },
    {
        "question": "What deficiency codes are documented for the WWTP (Facility 402) in project RM23-0697?",
        "options": [
            "A) W2 Code Compliance – Seismic, Severity 2; Y2 Location – Flood Plain, Severity 2",
            "B) W4 Code Compliance – Explosives Safety, Severity 2; Y3 Location – Site Characteristics, Severity 1",
            "C) Y2 Location – Flood Plain/Environmental Incompatibility, Severity 2; Z9 Inadequate Capacity – Supporting Systems, Severity 2",
            "D) W5 Code Compliance – Fire Codes, Severity 4; X2 Functional Criteria – Interior Configuration, Severity 2",
            "E) W7 Code Compliance – ATFP, Severity 2; X3 Functional Criteria – Building or Structure, Severity 2"
        ],
        "answer": "C) Y2 Location – Flood Plain/Environmental Incompatibility, Severity 2; Z9 Inadequate Capacity – Supporting Systems, Severity 2"
    },

    # RM23-0765 - POWER RELIABILITY & RESILIENCY UPGRADES (NISCEMI) - NAS SIGONELLA IT
    {
        "question": "What is the mission of the facility for project RM23-0765?",
        "options": [
            "A) NAS Sigonella requires reliable building infrastructure and must maintain and operate the installation electrical distribution system for transmission of electrical power",
            "B) Provide electrical power via expeditionary diesel engine-generator sets as two separate central power plants providing 90 percent of CLDJ's power",
            "C) Naval Radio Transmitter Facility (NRTF) Niscemi provides 24/7 mission-critical communication requirements for shore-to-shore and shore-to-ship HF, LF, VLF, and ELF communications for U.S., NATO, and coalition forces in AFRICOM, CENTCOM and EUCOM AORs",
            "D) Operate Building 64 as a power plant for the entirety of NAVSTA Rota Base",
            "E) Provide ordnance support and quick response team capability for VLS/CLS operations within the 5th Fleet AOR"
        ],
        "answer": "C) Naval Radio Transmitter Facility (NRTF) Niscemi provides 24/7 mission-critical communication requirements for shore-to-shore and shore-to-ship HF, LF, VLF, and ELF communications for U.S., NATO, and coalition forces in AFRICOM, CENTCOM and EUCOM AORs"
    },
    {
        "question": "What is the Region Readiness Support score and description for project RM23-0765?",
        "options": [
            "A) Score 3 – The RAC expires after 30 days; the installation has assumed undue risk and should discontinue use of the hangars per the Fire Chief",
            "B) Score 4 – Eliminates a failed Condition Rating (43) for critical utility and high-voltage electric line; the average current risk code for 20 components is 3.9, a NAVFAC red degradation index",
            "C) Score 2 – Project addresses major infrastructure capability gap identified in the 2020 NAS Sigonella Region Master Plan",
            "D) Score 4 – NRTF Niscemi is the linchpin of the U.S. military's next generation global communication system; Mobile User Objective System (MUOS) is estimated to cost $7B and is vitally important for security of U.S. interests, NATO, and the EU",
            "E) Score 4 – Project is in an exceptionally high impact area critical to Fleet/COCOM operations with no off-base infrastructure support"
        ],
        "answer": "D) Score 4 – NRTF Niscemi is the linchpin of the U.S. military's next generation global communication system; Mobile User Objective System (MUOS) is estimated to cost $7B and is vitally important for security of U.S. interests, NATO, and the EU"
    },

    # RM23-0785 - WATER SYSTEM F/P IMPROVEMENTS, PIERS, INFRASTRUCTURE - NAVSTA ROTA SP
    {
        "question": "What is the project scope for RM23-0785?",
        "options": [
            "A) Reconstruct Aircraft Parking Apron 1 with 13 inches of Portland Cement Concrete over six inches of base and make hydrological repairs to airfield drainage systems",
            "B) Replace backflow preventers for ship-to-shore water connections for all piers #1-#4; provide potable water storage tank for fire protection at Pier #3; repair Tanks #70 and #1980; and provide repairs to the NAVSTA Rota GAK plant",
            "C) Replace pumps, valves, piping, guide rails, wet well lining, security fencing, chemical eye wash, hatches, and electrical upgrades at lift stations",
            "D) Repair by replacement of the entire Building 64 envelope and reinforcement of concrete roof to comply with seismic and ATFP regulations",
            "E) Install fire suppression system and perform interior repair works in aircraft maintenance hangar Building 460"
        ],
        "answer": "B) Replace backflow preventers for ship-to-shore water connections for all piers #1-#4; provide potable water storage tank for fire protection at Pier #3; repair Tanks #70 and #1980; and provide repairs to the NAVSTA Rota GAK plant"
    },
    {
        "question": "What is the Region Severity score and description for project RM23-0785?",
        "options": [
            "A) Score 4 – Addresses Weapons Compound deficiencies necessary to meet NAVEUR's ordnance load plans; no waiver opportunity to increase NEW capacity without repairs",
            "B) Score 2 – Impact to mission readiness is significant if a seismic event incapacitates Power Plant Building 64 with compromise of electrical power supply",
            "C) Score 3 – Primary mission accomplished with significant workarounds; a single 12-inch water main feeds all piers and if this line fails, potable water supply to ships may be cut off with no alternative water source",
            "D) Score 4 – Multi-day work stoppage and overflow will cause environmental hazards; MBR, filters, and single pump sized to handle only 60% of installation's daily sanitary sewage output",
            "E) Score 3 – Primary mission accomplished with significant workarounds; Motors, valves, and piping unable to efficiently collect and transfer wastewater to treatment plants"
        ],
        "answer": "C) Score 3 – Primary mission accomplished with significant workarounds; a single 12-inch water main feeds all piers and if this line fails, potable water supply to ships may be cut off with no alternative water source"
    },

    # ST18-1369 - REPAIRS TO AIRFIELD PAVEMENT AND DRAINAGE - NAVSTA ROTA SP
    {
        "question": "What is the mission of the facility for project ST18-1369?",
        "options": [
            "A) Provide aircraft parking apron specifically in support of helicopter operations at NAVSTA Rota",
            "B) NS Rota airfield provides operational and logistic support for Commander Sixth Fleet, EUCOM, TRANSCOM, and CENTCOM forces; serves as a Primary Reception Airfield and Aerial Port of Debarkation as part of TRANSCOM's Global Enroute Strategy",
            "C) Provide adequate cargo staging area to temporarily store rolling stock, shipping containers, and equipment for intermodal operations",
            "D) Provide aircraft hangars with fire water suppression capability meeting UFC requirements",
            "E) Provide potable water storage, treatment, and pressure regulation for supply of potable water around Base"
        ],
        "answer": "B) NS Rota airfield provides operational and logistic support for Commander Sixth Fleet, EUCOM, TRANSCOM, and CENTCOM forces; serves as a Primary Reception Airfield and Aerial Port of Debarkation as part of TRANSCOM's Global Enroute Strategy"
    },
    {
        "question": "What is the PCI rating, RAC, and CWE for project ST18-1369?",
        "options": [
            "A) PCI 25, RAC II, CWE $19,016K",
            "B) PCI 33, RAC not specified, CWE $4,485K",
            "C) PCI 42, RAC II, CWE $28,001K",
            "D) PCI 42, RAC I, CWE $13,086K",
            "E) PCI 55, RAC II, CWE $28,001K"
        ],
        "answer": "C) PCI 42, RAC II, CWE $28,001K"
    },

    # ST19-0946 - REPAIRS TO AIRFIELD APRON #1 AND DRAINAGE - NAVSTA ROTA SP
    {
        "question": "What is the project scope for ST19-0946?",
        "options": [
            "A) Reconstruct runway sections, provide partial and full depth PCC repairs, reconstruct compass rose and VSTOL pad, and repair drainage in infield areas",
            "B) Repair about 63,300 SF area of Stand 55 including demolition of existing pavement and reconstruction of new concrete pavement",
            "C) Reconstruct Aircraft Parking Apron 1 (PA1) with 13 inches of PCC over six inches of base; provide localized slab replacement at APHA580; and make hydrological repairs to airfield drainage adjacent to Parking Apron 1",
            "D) Provide localized concrete slab replacement, partial and full depth repairs at Aircraft Parking Aprons 2, 4, 5, and 7",
            "E) Replace backflow preventers and provide potable water storage tank for fire protection at Pier 3"
        ],
        "answer": "C) Reconstruct Aircraft Parking Apron 1 (PA1) with 13 inches of PCC over six inches of base; provide localized slab replacement at APHA580; and make hydrological repairs to airfield drainage adjacent to Parking Apron 1"
    },
    {
        "question": "What is the Region Mission Alignment score and description for project ST19-0946?",
        "options": [
            "A) Score 3 – Project contributes to CNO directed basing decision and mission growth of homeported FNDF DDGs; closes a critical capability gap identified during the NAVSTA Rota IIG adjudication meeting",
            "B) Score 4 – NS Rota airfield provides operational and logistic support for C6F, EUCOM, TRANSCOM, and CENTCOM; serves as a Primary Reception Airfield and Deployment Operating Base for NATO missions; addresses capability gap ROTA-AIR08 in the Regional Master Plan",
            "C) Score 4 – Addresses Weapons Compound deficiencies providing infrastructure repairs necessary to meet NAVEUR's ordnance load plans",
            "D) Score 2 – Project addresses major infrastructure capability gap identified in the 2020 NAS Sigonella Region Master Plan",
            "E) Score 3 – Contributes to closing red capability gap as outlined in the Installation Development Plan for NAVSTA Rota; NEPMU-7 supports 3 COCOMs"
        ],
        "answer": "B) Score 4 – NS Rota airfield provides operational and logistic support for C6F, EUCOM, TRANSCOM, and CENTCOM; serves as a Primary Reception Airfield and Deployment Operating Base for NATO missions; addresses capability gap ROTA-AIR08 in the Regional Master Plan"
    },

    # ST21-0170 - PRIME POWER PLANT REPAIRS FY25 - CAMP LEMONNIER DJIBOUTI
    {
        "question": "What is the mission of the facility for project ST21-0170?",
        "options": [
            "A) Provide berthing accommodations for tenant commands and assigned personnel at CLDJ",
            "B) CLDJ has two separate central power plants, Prime Power 2 and Prime Power 3, which provide electrical power via expeditionary diesel engine-generator sets; the plants provide 90 percent of CLDJ's power and run at 90-100 percent capacity during summer months",
            "C) Provide 24/7 mission-critical communication requirements for U.S., NATO, and coalition forces operating in AFRICOM, CENTCOM, and EUCOM AORs",
            "D) Provide wastewater collection and treatment for the entire Naval Air Station Sigonella supporting U.S. and NATO forces",
            "E) Operate Building 64 as a central power plant for NAVSTA Rota Base"
        ],
        "answer": "B) CLDJ has two separate central power plants, Prime Power 2 and Prime Power 3, which provide electrical power via expeditionary diesel engine-generator sets; the plants provide 90 percent of CLDJ's power and run at 90-100 percent capacity during summer months"
    },
    {
        "question": "What is the Lead Proponent, COCOM, and CWE for project ST21-0170?",
        "options": [
            "A) Lead Proponent CNIC N3, COCOM EUCOM, CWE $9,174K",
            "B) Lead Proponent NAVFAC, COCOM AFRICOM, CWE $13,495K",
            "C) Lead Proponent CNIC N9, COCOM AFRICOM, CWE $17,454K",
            "D) Lead Proponent IWE, COCOM EUCOM, CWE $9,174K",
            "E) Lead Proponent NAE, COCOM EUCOM, CWE $13,495K"
        ],
        "answer": "B) Lead Proponent NAVFAC, COCOM AFRICOM, CWE $13,495K"
    },

]

    # Counts correct answers to questions
    # Requires that answers match exactly
    correct = 0
    # seed = 123
    # random.seed(seed)
    q_a_sample = random.sample(q_a_pairs, q_num)  # Shuffle questions with a fixed seed for reproducibility
    for pair in q_a_sample:
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

    accuracy = correct / len(q_a_sample)

    print(f"Pipeline accuracy: {accuracy:.02%}")


# ----------------------------------------------------
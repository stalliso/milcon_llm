import os
import json
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from .parser import FormTextExtractor, get_pds


def build_vectorstores():

    # Anchor all paths relative to this file, not the calling script
    base = Path.cwd()

    ########## PROJECT DATA SHEETS ##########
    directory = base / "docs/projects"
    proj_doc_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith('.pdf')]

    proj_docs_flat = []
    for doc_path in proj_doc_paths:
        try:
            boxes = FormTextExtractor(doc_path).extract_boxes()
            pds_dict = get_pds(boxes)
            project_id = list(pds_dict.keys())[0]
            project_data = pds_dict[project_id]
            doc = Document(
                page_content=json.dumps(project_data, indent=2),
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

    print(f'Total Documents Processed: {len(proj_docs_flat)}')

    ########## POM26 STRATEGY ##########
    directory = base / "docs/strategy/pom26"
    strat26_doc_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    strat26_docs_flat = [page for path in strat26_doc_paths for page in PyMuPDFLoader(path).load()]

    ########## POM28 STRATEGY ##########
    directory = base / "docs/strategy/pom28"
    strat28_doc_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]
    strat28_docs_flat = [page for path in strat28_doc_paths for page in PyMuPDFLoader(path).load()]

    ########## EMBEDDINGS ##########
    baseline_hfe = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )

    ########## CHUNKING AND VECTORSTORE CREATION ##########
    recursive_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    recursive_proj_docs    = recursive_splitter.transform_documents(proj_docs_flat)
    recursive_strat26_docs = recursive_splitter.transform_documents(strat26_docs_flat)
    recursive_strat28_docs = recursive_splitter.transform_documents(strat28_docs_flat)

    for i, chunk in enumerate(recursive_proj_docs, start=1):
        chunk.metadata['id'] = str(i)
    for i, chunk in enumerate(recursive_strat26_docs, start=1):
        chunk.metadata['id'] = str(i)
    for i, chunk in enumerate(recursive_strat28_docs, start=1):
        chunk.metadata['id'] = str(i)

    recursive_chunk_vectorstore_proj = Chroma.from_documents(
        documents=recursive_proj_docs,
        embedding=baseline_hfe,
        persist_directory=str(base / "databases/proj"),
        ids=[doc.metadata["id"] for doc in recursive_proj_docs]
    )
    recursive_chunk_vectorstore_strat26 = Chroma.from_documents(
        documents=recursive_strat26_docs,
        embedding=baseline_hfe,
        persist_directory=str(base / "databases/strat26"),
        ids=[doc.metadata["id"] for doc in recursive_strat26_docs]
    )
    recursive_chunk_vectorstore_strat28 = Chroma.from_documents(
        documents=recursive_strat28_docs,
        embedding=baseline_hfe,
        persist_directory=str(base / "databases/strat28"),
        ids=[doc.metadata["id"] for doc in recursive_strat28_docs]
    )

    
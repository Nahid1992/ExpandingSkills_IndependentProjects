"""
ingest.py — Knowledge Base Builder for MedRAG

Processes PDF documents into a searchable vector index for use in the RAG pipeline.

Pipeline:
    1. Load    — Reads PDF files from the data/ directory and extracts raw text
    2. Chunk   — Splits text into overlapping segments (500 tokens, 50 token overlap)
                 to preserve context across chunk boundaries
    3. Embed   — Converts each chunk into a dense vector representation using
                 HuggingFace sentence-transformers (all-MiniLM-L6-v2)
    4. Index   — Stores embeddings in a FAISS index for fast similarity search

Output:
    faiss_index/   — Saved locally; loaded at runtime by app.py for retrieval

Usage:
    python ingest.py

Note:
    Run this script once before starting the FastAPI server, or re-run
    whenever source documents in data/ are added or updated.
"""


from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def ingest(pdf_paths):
    docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("faiss_index")
    print(f"Ingested {len(chunks)} chunks")

if __name__ == "__main__":
    pdfs = ["data/EViT_MedIA_2023_PE.pdf", "data/Founation_ChestCT_MedIA_Journal.pdf", "data/Foundaiton_X_WACV_Conf_Main.pdf", "data/Foundation_CTPA_Generic_Slice_based_3D_Architecture_Main.pdf",
            "data/Foundation_CTPA_Generic_Slice_based_3D_Architecture_Supp.pdf", "data/Foundation_X_MedIA_Journal.pdf", "data/Foundation_X_WACV_Conf_Supp.pdf", "data/MLMI2021_Nahid_Seeking_an_Optimal_Approach_for_Computer-Aided Pulmonary_Embolism_Detection.pdf"]
    ingest(pdfs)

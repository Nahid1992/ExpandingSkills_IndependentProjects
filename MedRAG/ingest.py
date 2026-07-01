## Runs:
# python ingest.py

## This will build the knowledge base
# Load PDFs from local data/ directory and extract raw text;
# Chunk: split raw text into smaller pieces (500 words each with 50 words overlap);
# Embed and Store: converts chunks into vector using a model from HuggingFaceEmbeddings and stores the vector into FAISS index file;

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
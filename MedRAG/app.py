## Runs:
# ollama serve  # in first terminal
# uvicorn app:main --reload # in second terminal

## in third terminal
#  curl -X POST http://localhost:8000/query \
#   -H "Content-Type: application/json" \
#   -d '{"question": "What is Lock-Release Pretraining?"}'  

## Loads FAISS index from local directory;
## Conects to Ollama
## Assembles the full RAG chain and keeps it read

## query sent as questions
## embeds the questions into vector using the same HuggingFaceEmbeddings model
## searches 4 most similar chanks from FAISS
## inserts those chunks into prompt template as context
## sends the full and updated prompts to llama3.2 via Ollama
## returns the answer in terminal

## After docker is installed:
# terminal run: docker run -p 8000:8000 medrag

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

## For Local Run
# llm = OllamaLLM(model="llama3.2", temperature=0)

## For Docker
llm = OllamaLLM(
    model="llama3.2",
    temperature=0,
    base_url="http://host.docker.internal:11434"
)


prompt = ChatPromptTemplate.from_template("""
You are a medical AI assistant. Answer the question based only on the following context from medical research papers.
If the answer is not in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG chain
qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class Query(BaseModel):
    question: str

@app.post("/query")
def query(q: Query):
    answer = qa_chain.invoke(q.question)
    return {"answer": answer}

@app.get("/health")
def health():
    return {"status": "ok"}
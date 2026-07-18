import os
import sys
from pathlib import Path


def ensure_virtualenv() -> None:
    workspace_root = Path(__file__).resolve().parent.parent
    venv_python = workspace_root / ".venv" / "Scripts" / "python.exe"
    if venv_python.exists() and os.path.normcase(sys.executable) != os.path.normcase(str(venv_python)):
        os.execv(str(venv_python), [str(venv_python), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_virtualenv()

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

print("Loading PDF Document...")
pdf_path = BASE_DIR / "TechCorp_Official_Employee_Handbook.pdf"
loader = PyPDFLoader(str(pdf_path))
document = loader.load()

print("Chunking Document...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500 ,
    chunk_overlap = 50
)
chunks = text_splitter.split_documents(document)

print("Creating Vector Database...")
embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
vector_db = Chroma.from_documents(
    document = chunks,
    embedding = embeddings,
    persist_directory = "./Chroma_db"
)

retriever = vector_db.as_retriever(search_kwargs={"k": 2})

template = """
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Context: {context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash", temperature = 0)
def format_docs(docs):
    return "n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

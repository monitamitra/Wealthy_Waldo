from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import EdenAiEmbeddings

load_dotenv(".env")

# create cluster and add embeddings
loader = PyPDFLoader("data_sources/investing-101.pdf")
pages = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100, 
    length_function=len
)
docs = text_splitter.split_documents(pages)
embeddings = EdenAiEmbeddings(edenai_api_key=os.getenv("EDENAI_API_KEY"), provider="openai")
vector_db = FAISS.from_documents(docs, embeddings)
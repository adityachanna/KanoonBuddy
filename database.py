import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
pdf=PyPDFLoader('indian-penal-code.pdf')
pdf2=PyPDFLoader('3.pdf')
loader_all = MergedDataLoader(loaders=[pdf, pdf2])
docs_all = loader_all.load()
split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
newdoc=split.split_documents(docs_all)
embeddings = OllamaEmbeddings(
    model="snowflake-arctic-embed2:latest",
)
vectorstore = FAISS.from_documents(newdoc,embeddings)
vectorstore.save("vectorstore")
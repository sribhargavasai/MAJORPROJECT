from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import os

def build_vectordb():
    loader = DirectoryLoader("traffic_docs", glob="**/*.txt")

    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    db.save_local("urban_transport_faiss_index")
    print("✅ Vector DB created successfully!")

if __name__ == "__main__":
    build_vectordb()
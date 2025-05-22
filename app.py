
import os
import re
import fitz  # PyMuPDF
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st


# ------------------- Environment -------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyAckx-U3feW4dDEUGaWDDmKleATiPpJHqA"
os.environ["USER_AGENT"] = "UrbanTransportRAGBot/1.0"

# ------------------- LLM -------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ------------------- Embedding -------------------
@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings()

# ------------------- PDF Extraction -------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

# ------------------- Vector Store -------------------
def create_vectorstore_from_pdfs(uploaded_files):
    docs = []
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        docs.append(Document(page_content=text))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(split_docs, get_embedding())
    return vectordb

# ------------------- Agent Modules -------------------
def research_agent(query, vectordb, threshold=1.0):
    docs_scores = vectordb.similarity_search_with_score(query, k=4)
    relevant_docs = []

    for doc, score in docs_scores:
        if score <= threshold:
            relevant_docs.append(doc)

    if not relevant_docs:
        return None, 0

    return "\n".join([doc.page_content for doc in relevant_docs]), 1


def analyst_agent(context, query):
    prompt = ChatPromptTemplate.from_template(
        """
        You are an AI assistant that only answers using the provided context.
        Do not use any outside knowledge.
        If the question cannot be answered from the context, say "out of context".

        <context>
        {context}
        </context>

        Question: {input}
        """
    )
    formatted_prompt = prompt.format_messages(context=context, input=query)
    result = llm.invoke(formatted_prompt)
    return result.content


def main_agent(query, vectordb):
    blocked = ["politics", "religion", "relationships", "opinion", "news"]
    if any(x in query.lower() for x in blocked):
        return "❌ Sorry, that topic is restricted."

    context, found = research_agent(query, vectordb)
    if found:
        return analyst_agent(context, query)

    return "❌ No relevant information found in your uploaded PDFs. Please try another question or upload more documents."


# ------------------- URL Paragraph Extractor -------------------
def load_from_url(url):
    try:
        headers = {"User-Agent": os.getenv("USER_AGENT", "UrbanTransportRAGBot/1.0")}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        return [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
    except Exception as e:
        return [f"Error loading URL: {str(e)}"]


# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Urban Transport RAG", page_icon="🚌")
st.title("🚦 Urban Transportation Agentic RAG System")
st.markdown("AI-powered assistant for traffic, public transport & route planning based only on uploaded PDF knowledge.")

# Optional: URL Content Viewer
with st.expander("🔗 View External Web Content (Experimental)"):
    url_input = st.text_input("Enter URL to view urban transport article:")
    if st.button("Load & Display Article") and url_input:
        para = load_from_url(url_input)
        if para:
            st.success("Loaded paragraph:")
            st.markdown(f"**Paragraph:** {para[0]}")
        else:
            st.error("Failed to load paragraph.")

# Upload PDF files
st.subheader("📄 Upload Urban Transport PDFs")
uploaded_files = st.file_uploader("Upload PDF files for urban transport knowledge", type="pdf", accept_multiple_files=True)

if uploaded_files:
    vectordb = create_vectorstore_from_pdfs(uploaded_files)
    st.success("PDFs processed and vector database created!")
else:
    st.warning("Please upload at least one PDF to use the assistant.")
    vectordb = None

# Ask the bot
user_input = st.text_input("Ask your urban transportation-related question")

if user_input and vectordb:
    response = main_agent(user_input, vectordb)

    # Goodbye detection
    if re.search(r"\\b(bye|later|talk to you|take care|goodnight|farewell|see you)\\b", user_input.lower()):
        closing_responses = [
            "Safe travels!", "Catch you later!", "Drive safe!",
            "Until next time!", "Happy commuting!"
        ]
        st.success(closing_responses[hash(user_input) % len(closing_responses)])
    else:
        st.markdown(f"**Response:**\n{response}")

elif user_input:
    st.error("Please upload at least one PDF before asking.")

# Sidebar info
st.sidebar.header("Business Value")
st.sidebar.info("""
This Urban Transportation AI Agentic RAG system helps:
- Citizens plan smarter routes from uploaded data.
- City officials analyze transportation documents.
- Planners assess policy impact and efficiency.
- Blocks irrelevant and sensitive topics.

Answers strictly based on uploaded PDF knowledge.
""")

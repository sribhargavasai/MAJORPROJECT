import os
import re
import streamlit as st
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool, AgentType

# ------------------- Environment -------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyAckx-U3feW4dDEUGaWDDmKleATiPpJHqA"
os.environ["USER_AGENT"] = "UrbanTransportRAGBot/1.0"

# ------------------- LLM -------------------
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ------------------- External Tools -------------------
def get_real_time_traffic(city):
    return f"Traffic in {city} is currently moderate with delays on major routes."

def get_public_transport_info(city):
    return f"Metro and bus services in {city} are operational with 90% on-time performance."

resource_tools = [
    Tool(name="RealTimeTraffic", func=lambda x: get_real_time_traffic("Hyderabad"), description="Get real-time traffic info"),
    Tool(name="PublicTransport", func=lambda x: get_public_transport_info("Hyderabad"), description="Get public transport efficiency")
]

external_agent = initialize_agent(
    tools=resource_tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)

# ------------------- Vector Store -------------------
@st.cache_resource
def get_embedding():
    return HuggingFaceEmbeddings()

from langchain.text_splitter import RecursiveCharacterTextSplitter

def create_vectorstore_from_pdfs(uploaded_files):
    docs = []
    for file in uploaded_files:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        docs.append(Document(page_content=text))
    
    # ✅ Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    
    vectordb = FAISS.from_documents(split_docs, get_embedding())
    return vectordb


# ------------------- Agents -------------------
def research_agent(query, vectordb, threshold=0.7):
    docs_scores = vectordb.similarity_search_with_score(query)
    relevant_docs = [doc for doc, score in docs_scores if score <= threshold]
    if not relevant_docs:
        return None, 0
    return "\n".join([doc.page_content for doc in relevant_docs]), 1

def analyst_agent(response):
    prompt = f"You are an urban transportation assistant. Simplify and validate the following content. Ensure it is clear and accurate for a citizen or planner:\n\n{response}"
    analysis = llm.invoke(prompt)
    return analysis.content

def solution_agent(query):
    if "route" in query.lower() or "plan" in query.lower():
        return "Step-by-step Travel Plan:\n1. Enter source and destination.\n2. Get suggested modes.\n3. Consider traffic info.\n4. Choose optimal route."
    return "Try asking about route planning, congestion solutions, or multimodal options."

def main_agent(query, vectordb):
    blocked = ["politics", "religion", "relationships", "opinion", "news"]
    if any(x in query.lower() for x in blocked):
        return "Sorry, that topic is restricted."

    context, found = research_agent(query, vectordb)
    if found:
        return analyst_agent(f"Domain Knowledge:\n{context}")

    elif "how to" in query.lower() or "steps" in query.lower():
        return solution_agent(query)

    elif any(term in query.lower() for term in ["traffic", "bus", "metro", "train"]):
        return external_agent.run(query)

    else:
        return "We couldn't find relevant information in your uploaded documents. Try rephrasing your query or upload more relevant PDFs."

# ------------------- URL Paragraph Extractor -------------------
def load_from_url(url):
    try:
        headers = {"User-Agent": os.getenv("USER_AGENT", "UrbanTransportRAGBot/1.0")}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        for para in paragraphs:
            text = para.get_text(strip=True)
            if text:
                return [text]
        return ["No readable paragraph found."]
    except Exception as e:
        return [f"Error loading URL: {str(e)}"]

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Urban Transport RAG", page_icon="🚌")
st.title("🚦 Urban Transportation Agentic RAG System")
st.markdown("AI-powered assistant for traffic, public transport & route planning")

# Optional: URL Content Viewer
with st.expander("🔗 View External Web Content (Experimental)"):
    url_input = st.text_input("Enter URL to view urban transport article:")
    if st.button("Load & Display Article") and url_input:
        para = load_from_url(url_input)
        if para:
            st.success("Loaded first paragraph:")
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
    if re.search(r"\b(bye|later|talk to you|take care|goodnight|farewell|see you)\b", user_input.lower()):
        closing_responses = [
            "Safe travels!", "Catch you later!", "Drive safe!",
            "Until next time!", "Happy commuting!"
        ]
        st.success(closing_responses[hash(user_input) % len(closing_responses)])
    else:
        if response == "Sorry, that topic is restricted.":
            st.markdown(f"**Response:** {response}")
        else:
            simplified_response = analyst_agent(response)
            st.markdown(f"**Response:**\n{simplified_response}")
elif user_input:
    st.error("Please upload at least one PDF before asking.")

# Sidebar info
st.sidebar.header("Business Value")
st.sidebar.info("""
This Urban Transportation AI Agentic RAG system helps:
- Citizens plan smarter routes avoiding traffic.
- City officials assess public transport efficiency.
- Planners analyze mobility trends from uploaded documents.
- Enables fallback answers for unindexed queries.
- Blocks irrelevant and sensitive topics.

Powered by RAG + Tools + Agents
""")

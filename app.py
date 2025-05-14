import streamlit as st
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import re

# ---------------- Setup ----------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyAckx-U3feW4dDEUGaWDDmKleATiPpJHqA"


# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# ---------------- Resource Agent ----------------
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
)

# ---------------- Vector Store (RAG) ----------------
@st.cache_resource
def load_vectordb():
    embedding = HuggingFaceEmbeddings()
    return FAISS.load_local("urban_transport_faiss_index", embedding, allow_dangerous_deserialization=True)

# ---------------- Research Agent ----------------
def research_agent(query, vectordb, threshold=0.7):
    docs_scores = vectordb.similarity_search_with_score(query)
    relevant_docs = [doc for doc, score in docs_scores if score <= threshold]
    if not relevant_docs:
        return None, 0
    return "\n".join([doc.page_content for doc in relevant_docs]), 1

# ---------------- Analyst Agent ----------------
def analyst_agent(response):
    prompt = f"You are an urban transportation assistant. Simplify and validate the following content. Ensure it is clear and accurate for a citizen or planner:\n\n{response}"
    analysis = llm.invoke(prompt)
    return analysis.content

# ---------------- Solution Developer Agent ----------------
def solution_agent(query):
    if "route" in query.lower() or "plan" in query.lower():
        return "Step-by-step Travel Plan:\n1. Enter source and destination.\n2. Get suggested modes.\n3. Consider traffic info.\n4. Choose optimal route."
    return "Try asking about route planning, congestion solutions, or multimodal options."

# ---------------- Main Agent ----------------
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
        return external_agent.invoke(query)

    else:
        return "We couldn't find relevant information in our knowledge base. This might be:\n- Too specific or out of domain\n- Not covered in our 20+ urban documents\n\nTry rephrasing or contact a local expert."

# ---------------- URL Content Loader ----------------
def load_from_url(url):
    try:
        loader = WebBaseLoader(url)
        return loader.load()
    except Exception as e:
        return []

# ---------------- Streamlit UI ----------------
st.title("Urban Transportation Agentic RAG System")
st.markdown("AI-powered assistant for traffic, public transport & route planning")

# Optional: URL Content Viewer
with st.expander("🔗 View External Web Content (Experimental)"):
    url_input = st.text_input("Enter URL to view urban transport article:")
    if st.button("Load & Display Article") and url_input:
        docs = load_from_url(url_input)
        if docs:
            st.success(f"Loaded {len(docs)} paragraph(s) from the webpage.")
            for i, doc in enumerate(docs):
                st.markdown(f"**Paragraph {i+1}:** {doc.page_content}")
        else:
            st.error("Failed to load content from URL.")

user_input = st.text_input("Ask your urban transportation-related question")

if user_input:
    vectordb = load_vectordb()
    response = main_agent(user_input, vectordb)

    # Handling Goodbye Messages with Dynamic Content
    if re.search(r"\b(bye|later|talk to you|take care|goodnight|farewell|see you)\b", user_input.lower()):
        closing_responses = [
            "Safe travels!",
            "Catch you later!",
            "Drive safe!",
            "Until next time!",
            "Happy commuting!"
        ]
        st.success(closing_responses[hash(user_input) % len(closing_responses)])
    else:
        if response == "Sorry, that topic is restricted.":
            st.markdown(f"**Response:** {response}")
        else:
            simplified_response = analyst_agent(response)
            st.markdown(f"**Response:**\n{simplified_response}")

# ---------------- Business Value ----------------
st.sidebar.header("Business Value")
st.sidebar.info("""
This Urban Transportation AI Agentic RAG system helps:
- Citizens plan smarter routes avoiding traffic.
- City officials assess public transport efficiency.
- Planners analyze mobility trends from 20+ urban documents.
- Enables smart fallback answers for unindexed queries.
- Blocks irrelevant and sensitive topics.

Powered by RAG + Tools + Agents
""")

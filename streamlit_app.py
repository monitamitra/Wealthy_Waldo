import streamlit as st
from dotenv import load_dotenv
import os
from langchain import hub
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.agents import tool
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_cohere import ChatCohere


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

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
# embeddings = EdenAiEmbeddings(edenai_api_key=os.getenv("EDENAI_API_KEY"), provider="openai")
vector_db = FAISS.from_documents(docs, embeddings)

LLM_API_KEY = os.getenv("GEMINI_API_KEY")

retriever = vector_db.as_retriever()
retriever_tool = create_retriever_tool(
    retriever,
    "Knowledge Base",
    """Search for information about different asset classes in investment portfolios. 
    For generating a specific investment portfolio, you must use this tool!"""
)

tools = [retriever_tool]

st.title("🦜🔗 Wealthy Waldo: Your Investment Planning Assistant")

prompt_str_template = """your name is Wealthy Waldo. You are an investment planning assistant who generates a 
    personalized and specific investment portfolio for a user based on the characteristics of their profile. 
    Given a user with a {risk_tolerance} risk tolerance, {investment_goal} investment goal, 
    and a {investment_horizon} investment horizon, and considering the current market data and respective 
    news for specific asset classes that you feel are necessary, your job is to generate a diversified 
    investment portfolio that aligns with the user's preferences. Prioritize assets with {investment_style} investment style characteristics."""

prompt_str = ""

def generate_response():
    llm = ChatCohere(cohere_api_key=os.getenv("COHERE_API_KEY"))
    prompt = [ChatPromptTemplate.from_messages([
        ("system", prompt_str),
        ("user", "{input}"),
        ("placeholder", "{agent_scratchpad}")], 
    tools = tools, 
    tool_names = ["Knowledge Base"]
)]
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": "generate an investment plan for me.", })
    st.info(result)

with st.form('my_form'):
    st.info('Hello! I am Wealthy Waldo! What can I do to make you wealthy today?')
    risk_tolerance_option = st.select_slider("Risk Tolerance", options = [ "Conservative", "Moderate", "Aggressive"])
    investment_goals = st.text_area("What are your short-term or long-term goals?")
    investment_horizon_option = st.select_slider("Investment Horizon", options = ["Short Term (few months to 3 years)", 
                                                                                  "Medium Term (5-10 years)", 
                                                                                  "Long Term (at least 10 years)"])
    investment_styles_option = st.select_slider("Investment Styles", options = ["Passive", "Active"])
    submitted = st.form_submit_button('Submit')
    if submitted:
        prompt_str = prompt_str_template.format(risk_tolerance = risk_tolerance_option, investment_goal = investment_goals, 
                               investment_horizon = investment_horizon_option, investment_style = investment_styles_option)
        generate_response()

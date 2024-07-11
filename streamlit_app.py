import streamlit as st
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAI


load_dotenv(".env")
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# turn text into embeddings and load into vector store 
loader = TextLoader("asset_class.md")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30, 
    length_function=len
)
doc_splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
db = FAISS.from_documents(doc_splits, embeddings)

# define tools for langchain agent to use => tavily to search internet and chromadb to store vector embeddings
vector_store_retriever = db.as_retriever()
retriever_tool = create_retriever_tool(
    vector_store_retriever,
    "Asset class knowledge base",
    """Search for specific information about different asset classes in investment portfolios. 
    For generating a specific investment portfolio, you must use this tool!"""
)

web_search_tool = TavilySearchResults(max_results = 4)
web_search_tool.description = """find relevant information fromt the internet needed to contruct the user's 
    investment portfolio"""

tools = [web_search_tool, retriever_tool]

# prompt template telling the llm what it does 
prompt_str_template = """your name is Wealthy Waldo. You are an investment planning assistant who generates a 
    personalized and specific investment portfolio for a user based on the characteristics of their profile. 
    Given a user with a {risk_tolerance} risk tolerance, {investment_goal} investment goal, 
    and a {investment_horizon} investment horizon, and considering the current market data and respective news for 
    specific asset classes that you feel are necessary, your job is to generate a diversified investment portfolio 
    that aligns with the user's preferences. Prioritize assets with {investment_style} investment style 
    characteristics.First use tools to search the internet for a general investment portfolio plan and then 
    for each asset class use your asset class knowledge base to search for specific information about 
    different asset classes in investment portfolios. You can use necessary tools to accompish your task.
    The output should be like this: 
    **Overall Asset Allocation :
    * asset_class_1: allocation_1%
    * asset_class_2: allocation_2%
    * ... and so on for all asset classes

    **Detailed Asset Class Breakdowns => using Vector Store**
    **For each asset class retrieved:
  * Query the vector store to find information on relevant subcategories and investment options specific to that asset class
  and user's specific investment goals, investment horizon, and risk_tolerance. 
  * Analyze the retrieved data using historical performance, risk profiles, etc  based on the asset class type.
  * Based on this analysis and user input, recommend specific allocations for subcategories within the asset class. 
  * Explain the rationale behind the allocation percentages for each subcategory."""

prompt_str = ""

# streamlit formatting
st.title("🦜🔗 Wealthy Waldo: Your Investment Planning Assistant 💸")

# initiates agent action to generate portfolio
def generate_response():
    llm = OpenAI(openai_api_key=os.getenv("OPEN_AI_API_KEY"), temperature = 0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", prompt_str),
        ("user", "{input}")])
    
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    result = agent_executor.invoke({"input": "generate an personalized investment portfolio for me." })
    st.info(result.get("output"))

# formats prompt template for langchain agent according to user investment profile
with st.form('my_form'):
    st.info('Hello! I am Wealthy Waldo! What can I do to make you wealthy today?')

    risk_tolerance_option = st.select_slider("Risk Tolerance", options = [ "Conservative", "Moderate", "Aggressive"])
    investment_goals = st.text_area("What are your short-term or long-term goals?")
    investment_horizon_option = st.select_slider("Investment Horizon", options = ["Short Term (few months to 3 years)", 
                                                                                  "Medium Term (5-10 years)", 
                                                                                  "Long Term (at least 10 years)"])
    investment_styles_option = st.selectbox("Investment Styles", options = ["Passive", "Active"])
    submitted = st.form_submit_button('Submit')
    if submitted:
        prompt_str = prompt_str_template.format(risk_tolerance = risk_tolerance_option, investment_goal = investment_goals, 
                               investment_horizon = investment_horizon_option, investment_style = investment_styles_option)
        generate_response()

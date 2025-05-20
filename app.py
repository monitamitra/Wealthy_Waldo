import streamlit as st
import os
import yfinance as yf
import nltk
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import TextLoader
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document

load_dotenv(find_dotenv())
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng') 

def generate_response(human_prompt):
    # Load document if file is uploaded
    if human_prompt is not None:
        main_agent = create_agent()
        response = main_agent.invoke({"messages": HumanMessage(human_prompt)})
        return response["messages"][-1].content

def create_agent():
        tools = [vectordb_tool, websearch_tool, asset_performance_tool]

        llm = ChatOpenAI(temperature = 0)

        system_prompt = SystemMessage("""
You are Wealthy Waldo, an investment planning assistant that generates personalized portfolios 
based on user inputs: risk tolerance, investment goal, investment horizon, and investment style.

Follow this reasoning process:

1. Use the **Asset_class_knowledge_base** (vectordb_tool) to identify suitable asset classes 
   and general allocation strategies for the user profile.
2. Use the **websearch_tool** to gather current news or trends for each recommended asset class.
3. Use the **asset_performance_tool** to fetch real-time price and daily change data 
   (e.g., for tickers like 'VTI', 'BND', or 'QQQ') to support or refine your recommendations.

In your output, provide:
- A high-level asset allocation breakdown (e.g., 60%% stocks, 40%% bonds) based on user input and knowledge base.
- Supporting rationale that incorporates both historical insights and live data.
- Optional: Highlight any market conditions that may impact the user's portfolio or require caution.

Example Output:
**Overall Asset Allocation**
- Bond ETFs: 20%
- Stock ETFs: 70%
- REITs: 10%

**Rationale**
- Bond ETFs (e.g., BND) are suitable for moderate risk tolerance; currently trading at $74.10 (-0.25% today).
- Stock ETFs (e.g., VTI) align with your long-term goal and show positive daily growth (+1.12%).
- REITs diversify your portfolio and historically perform well in medium-term horizons.

Use concise, actionable language. Always explain your reasoning with reference to retrieved data or live performance.
""")
        
        return create_react_agent(llm, tools, state_modifier=system_prompt)

@tool
def vectordb_tool():
        # loader = TextLoader("knowledge_base.md")
        markdown_path = "knowledge_base.md"
        loader = UnstructuredMarkdownLoader(markdown_path)
        data = loader.load()
        documents = "\n\n".join([doc.page_content for doc in data])
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
        texts = text_splitter.create_documents(documents)
        # Select embeddings
        embeddings = OpenAIEmbeddings()
        # Create a vectorstore from documents
        db = FAISS.from_documents(texts, embeddings)
        # Create retriever interface
        retriever = db.as_retriever()
        return create_retriever_tool (
        retriever,
        "Asset_class_knowledge_base",
    """Search for specific information about what type of investments to include 
    in the user's personalized investment portfolio based on their risk tolerance, 
    investment goal, investment horizon, and investment style.""")

@tool
def websearch_tool():
    web_search_tool = TavilySearchResults(max_results = 4)
    web_search_tool.description = """find relevant information and/or news about each specific asset 
    class in the user's investment portfolio from the internet to advise the user on 
    constructing their investment portfolio."""
    return web_search_tool

@tool
def asset_performance_tool(ticker: str) -> str:
    """Fetches current price and daily change % for a given asset ticker."""
    try:
        stock = yf.Ticker(ticker)
        price = stock.info['regularMarketPrice']
        change = stock.info['regularMarketChangePercent']
        return f"{ticker} is trading at ${price:.2f} ({change:+.2f}%) today."
    except:
        return f"Could not fetch performance data for {ticker}."

asset_performance_tool.description = (
    "Get current market price and daily %% change for a financial asset (e.g., ETF or stock ticker like 'VTI')."
)

# Page title
st.set_page_config(page_title='💸 Wealthy Waldo 🤑')
st.title("🤑🔗 Wealthy Waldo: Your Investment Planning Assistant 💸")
st.info('Hello! I am Wealthy Waldo! What can I do to make you wealthy today?')

# Form input 
result = []
with st.form('myform', clear_on_submit=True):
    risk_tolerance = st.select_slider("Risk Tolerance", options = [ "Conservative", "Moderate", "Aggressive"])
    investment_goal = st.text_area("What are your short-term or long-term goals?")
    investment_horizon = st.select_slider("Investment Horizon", options = ["Short Term (few months to 3 years)", 
                            "Medium Term (5-10 years)", "Long Term (at least 10 years)"])
    investment_style = st.selectbox("Investment Styles", options = ["Passive", "Active"])
    
    submitted = st.form_submit_button('Submit', disabled = not(risk_tolerance or 
                    investment_goal or investment_horizon or investment_style))

    if submitted:
        with st.spinner('Generating your investment plan...'):
            human_template = """
            Generate an personalized investment portfolio for me with a {risk_tolerance} 
            risk tolerance, {investment_goal} investment goal, and a {investment_horizon} 
            investment horizon."""
            
            human_prompt = human_template.format(risk_tolerance = risk_tolerance, 
                        investment_goal = investment_goal, investment_horizon = 
                        investment_horizon, investment_style = investment_style)
            response = generate_response(human_prompt)
            result.append(response)
                     

if len(result):
    st.info(response)
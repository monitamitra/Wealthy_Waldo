import streamlit as st
import yfinance as yf
from langgraph.prebuilt import create_react_agent
import nltk
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv, find_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.agents import create_openai_functions_agent, AgentExecutor

load_dotenv(find_dotenv())
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng') 

def generate_response(human_prompt, uploaded_file=None):
    # Load document if file is uploaded
    if human_prompt is not None:
        main_agent = create_agent(uploaded_file)
        response = main_agent.invoke({"input": human_prompt})
        return response["messages"][-1].content


def create_agent(uploaded_file=None):
        tools = [vectordb_tool(), websearch_tool(), asset_performance_tool]

        # if user has uploaded file, create tool
        if uploaded_file is not None:
            docs = [uploaded_file.read().decode()]
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
            texts = text_splitter.create_documents(docs)
            # Select embeddings
            embeddings = OpenAIEmbeddings()
            # Create a vectorstore from documents
            db = FAISS.from_documents(texts, embeddings)
            # Create retriever interface
            retriever = db.as_retriever()
            # create tool
            user_notes_tool = create_retriever_tool(
            retriever,
            name="user_notes_tool",
            description="Retrieve personal investment notes and preferences from the user's uploaded document"
        )
            # append tool to tool list
            tools.append(user_notes_tool)


        llm = ChatOpenAI(temperature = 0.3)

        system_prompt = """
You are Wealthy Waldo, a sophisticated AI investment assistant.

Your task is to generate a personalized investment portfolio based on:
- Risk tolerance
- Investment goal
- Investment horizon
- Investment style
- User-uploaded notes (if provided)

You have access to the following tools:
1. `vectordb_tool` — to retrieve recommended asset allocation strategies.
2. `websearch_tool` — to gather recent news or market trends.
3. `asset_performance_tool` — to get current prices and daily performance of tickers (e.g., VTI, BND).
4. `user_notes_tool` — to extract personal preferences (e.g., avoid crypto, ESG-only, long-term focus, no tech).

---

🎯 **Format your full output like this**:

### 💼 Personalized Investment Plan

**📊 Overall Allocation**
List each asset class and % allocation. Only include assets justified by tools or user preferences.

---

**📈 Rationale**

For each asset class:
- Explain its role in the strategy (stability, growth, diversification)
- Mention how it fits the user’s risk level, goal, and time horizon
- Use real data from `asset_performance_tool` (e.g., “BND is trading at $71.83 (+0.23%) today”)
- Suggest 1–2 specific ETFs (like VTI, AGG, XLK) and explain why

---

**📝 User Note Integration**
If user notes are uploaded:
- Pull in any explicit preferences or exclusions (e.g., “Avoid crypto and tech” or “Include ESG funds only”)
- Respect constraints and clearly explain how they were applied
- Quote their note if helpful for transparency

---

**🌐 Market Context**
- Include 2–3 key insights from `websearch_tool`
- Tie trends to the current recommendation (e.g., “Rising bond yields make BND attractive for income-seekers.”)

---

**✅ Next Steps**
- Recommend a SMART plan:
  - Rebalance every 6–12 months
  - Monitor key sector/ETF performance
  - Adjust based on changes in interest rates, inflation, or retirement timeline

---

🧠 Be concise, confident, and data-grounded. Do NOT guess or make assumptions.
Only recommend assets when supported by tool data or user instructions.
"""

        return create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt)


def vectordb_tool():
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
        return create_retriever_tool(
        retriever,
        name="vectordb_tool",
        description="""Search for specific information about what type of investments to include 
    in the user's personalized investment portfolio based on their risk tolerance, 
    investment goal, investment horizon, and investment style."""
    )

def websearch_tool():
    return TavilySearchResults(max_results = 4,
                    description = """find relevant information and/or news about each specific asset 
    class in the user's investment portfolio from the internet to advise the user on 
    constructing their investment portfolio.""")

@tool
def asset_performance_tool(ticker: str) -> str:
    """Get current market price and daily %% change for a financial asset (e.g., ETF or stock ticker like 'VTI')."""
    print("[CALLED] get_asset_performance")
    
    try:
        stock = yf.Ticker(ticker)
        price = stock.info['regularMarketPrice']
        change = stock.info['regularMarketChangePercent']
        return f"{ticker} is trading at ${price:.2f} ({change:+.2f}%) today."
    except:
        return f"Could not fetch performance data for {ticker}."

# Page title
st.set_page_config(page_title='💸 Wealthy Waldo 🤑')
st.title("🤑🔗 Wealthy Waldo: Your Investment Planning Assistant 💸")
st.info('Hello! I am Wealthy Waldo! What can I do to make you wealthy today?')

# Form input 
result = []
uploaded_file = st.file_uploader("📄 Optionally upload your financial notes (Markdown or text)", type=["txt", "md"])
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
            response = generate_response(human_prompt, uploaded_file)
            result.append(response)
                     

if len(result):
    st.info(response)
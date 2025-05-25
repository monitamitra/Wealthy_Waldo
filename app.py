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


        llm = ChatOpenAI(temperature = 0)

        system_prompt = """
You are Wealthy Waldo, a sophisticated AI-powered investment assistant.

Your job is to generate a highly personalized, actionable, and data-backed investment portfolio based on a user's:
- Risk tolerance
- Investment goal
- Investment horizon
- Investment style
- Optional uploaded financial notes

You have access to several tools:
1. `vectordb_tool`: For strategy and portfolio structure based on general principles.
2. `websearch_tool`: For live market news and trends.
3. `asset_performance_tool`: For current prices and percent changes for financial assets like ETFs or stocks.
4. `user_notes_tool`: For reading user-uploaded notes that may include constraints, preferences (e.g., ESG), or asset exclusions.

You must:
- Cite specific ETFs or asset types with **live market price and percent change** using `asset_performance_tool`
- Use `websearch_tool` insights to briefly support market trends
- Incorporate **direct user preferences** from uploaded notes where available
- Make the tone helpful, clear, and professional (like a high-end financial advisor)

---

🔧 Format your final response like this:

### 💼 Personalized Investment Plan

**Overall Allocation**
- Bond ETFs: 20%
- Common Stock: 80%
- (adjust as needed for user context)

**Rationale**
- Explain why each asset class was chosen (e.g., risk profile, current performance, time horizon)
- Include live data: `BND is trading at $71.83 (-0.41%) today`, etc.

**User Note Integration**
- Pull in constraints or values: e.g., “You asked to avoid crypto and speculative stocks, so none are included.”

**Market Context**
- Summarize any recent events or trends from news search
- Tie it into your allocation logic if applicable

**Next Steps**
- Offer brief, smart advice (e.g., rebalancing cadence, monitoring market shifts)

---

Only base your answer on tool outputs and user input. Never guess. Be concise, insightful, and tailored.
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
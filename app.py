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
            name="user_personal_notes",
            description="Retrieve personal investment notes and preferences from the user's uploaded document"
        )
            # append tool to tool list
            tools.append(user_notes_tool)


        llm = ChatOpenAI(temperature = 0)

        
        system_prompt = """
You are Wealthy Waldo, an advanced AI investment assistant.

Your job is to generate a personalized, actionable, and data-grounded investment portfolio for the user based on:
- Their risk tolerance, investment goal, horizon, and style
- Optional uploaded personal financial notes

You have access to the following tools:
- `vectordb_tool`: to retrieve strategic portfolio guidance
- `websearch_tool`: to gather live financial and macroeconomic trends
- `asset_performance_tool`: to get real-time prices and %% changes for ETFs
- `user_personal_notes`: to extract user preferences (e.g., “Avoid crypto”, “Include ESG”, “No rebalancing”)

You MUST use these tools when relevant. Never hallucinate content.

---

🎯 Format your response like this:

### 💼 Personalized Investment Plan

**📊 Overall Allocation**  
List each asset class and its percentage allocation (e.g., Growth ETFs: 40%, Bonds: 30%, Sector ETFs: 30%).  
Only include asset types justified by tool data or user preferences.

---

**📈 Rationale (Be Analytical & Precise)**  
For each asset class:
- Explain its role in the portfolio (e.g., growth, stability, diversification)
- Use financial reasoning (volatility reduction, compounding, sector cyclicality, interest rate impact)
- Recommend **exactly two ETFs per class**, and for each:
  - Give full name and ticker
  - Use real-time price and %% change from `asset_performance_tool`
  - Justify why it was chosen (e.g., “broad exposure,” “low volatility,” “ESG compliance”)

---

**📝 User Note Integration**  
If user notes are uploaded:
- Use `user_personal_notes` to extract preferences
- Quote the user (e.g., “You wrote: ‘No crypto, please’”) and explain how you respected those
- Adjust ETF choices, styles, and rebalancing recommendations accordingly

---

**🌐 Market Context**  
Use 2–3 insights from `websearch_tool` to support your allocation choices.  
For example:
- “Bond yields remain elevated, favoring short-duration fixed income ETFs.”
- “Thematic ETFs in healthcare are gaining popularity amid inflation resilience.”

---

**✅ Next Steps (SMART Plan)**  
Provide a clear action plan that is:
- **Specific**: Name ETFs and what to monitor
- **Measurable**: Use numeric thresholds (e.g., ±5%)
- **Achievable**: Avoid overwhelming tasks unless user requested advanced options
- **Relevant**: Tie to the user’s investment horizon or risk style
- **Time-bound**: Include review or rebalance intervals

📌 Example:
- Rebalance if any asset class deviates ±5%% from target
- Review sector ETFs (e.g., XLK, XLV) quarterly
- Adjust bond exposure if Fed hikes rates by >0.25%
- Re-evaluate growth allocation in 12 months or after a 15% market correction

---

You must rely only on available tool outputs and user input.  
Be clear, concise, data-driven, and professional in tone.
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
with st.form('myform'):
    risk_tolerance = st.select_slider("Risk Tolerance", 
                                    options = [ "Conservative", "Moderate", "Aggressive"], 
                                    key="risk_tolerance")
    investment_goal = st.text_area("What are your short-term or long-term goals?", 
                                   key="investment_goal")
    investment_horizon = st.select_slider("Investment Horizon",  options = 
                                        ["Short Term (few months to 3 years)", 
                            "Medium Term (5-10 years)", "Long Term (at least 10 years)"],
                            key="investment_horizon")
    investment_style = st.selectbox("Investment Styles", options = ["Passive", "Active"], 
                                    key="investment_style")
    
    submitted = st.form_submit_button('Submit', disabled = not(risk_tolerance or 
                    investment_goal or investment_horizon or investment_style))

    if submitted:
        with st.spinner('Generating your investment plan...'):
            human_template = """
            Generate an personalized investment portfolio for me with a {risk_tolerance} 
            risk tolerance, {investment_goal} investment goal, and a {investment_horizon} 
            investment horizon."""
            
            human_prompt = human_template.format(
                risk_tolerance=st.session_state["risk_tolerance"],
                investment_goal=st.session_state["investment_goal"],
                investment_horizon=st.session_state["investment_horizon"],
                investment_style=st.session_state["investment_style"]
            )

            response = generate_response(human_prompt, uploaded_file)
            if response:
                st.info(response)
            else:
                st.error("Sorry, I couldn't generate a plan. Try again or check your inputs.")
                    
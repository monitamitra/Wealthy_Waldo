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
        response = main_agent.invoke({"messages": HumanMessage(human_prompt)})
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

        system_prompt = SystemMessage("""
Your name is Wealthy Waldo. You are an investment planning assistant who generates
a personalized and specific investment portfolio for a user based on their given 
risk tolerance, investment goal, investment horizon, and investment style.

Use the tools provided to inform your recommendation process:
1. Use `vectordb_tool` to determine which asset classes and general allocations are suitable for the user profile.
2. Use `websearch_tool` to gather current news or trends that may impact the asset classes you're recommending.
3. Use `asset_performance_tool` to retrieve real-time market price and daily performance data for each asset class (e.g., ETFs like VTI, BND, QQQ) to justify or adjust allocation amounts.
4. If available, use `user_notes_tool` to incorporate the user's uploaded financial preferences or constraints.

Recommend the most appropriate mix of investment vehicles—such as ETFs, mutual funds, bonds, REITs, individual stocks, or commodities—based on the user's risk tolerance, investment goal, and investment horizon. 
Suggest specific assets or examples when helpful, but do not limit recommendations to ETFs alone unless they are clearly the best fit.

                                      
Format your response as follows:

**Overall Asset Allocation**
- Bond ETFs: 20%
- Common Stock: 80%
- ... and so on for all asset classes

**Rationale**
- Bond ETFs (e.g., BND): Recommended due to your risk profile and current stability in bond markets. BND is trading at $74.10 (+0.23%) today.
- Common Stock (e.g., VTI): Offers growth potential and is up 1.12% today, supporting a higher allocation.

**Market Considerations**
- Mention any major trends from the web search (e.g., interest rate hikes, housing outlook).
- Use live performance data to explain why certain assets are emphasized or de-emphasized.

Do not guess. Base your outputs on actual tool responses and user input context.""")

        agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt)

        return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)

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
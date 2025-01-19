import streamlit as st
import nltk
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
        vectordb_tool = create_vectordb()
        web_search_tool = create_websearch_tool()
        tools = [vectordb_tool, web_search_tool]

        llm = ChatOpenAI(temperature = 0)

        system_prompt = SystemMessage("""Your name is Wealthy Waldo. You are an investment planning assistant who generates
        a personalized and specific investment portfolio for a user based on their given 
        risk tolerance, investment goal, investment horizon, and investment style. First use 
        the vectordb_tool to decide what types of assets and their respective allocations
        the user should incorporate into their portfolio. For each asset class that you decided 
        to include in the user's investment portfolio, use the web_search_tool to search any 
        news or information related to the respective assets. 
    
        Example Output: 
        **Overall Asset Allocation :
        * Bond ETFs: 20%
        * Common Stock: 80%
        * ... and so on for all asset classes
        - explanation: Bond ETFs is generally suited for people with your 
            investment style. According to recent news, common stock has been on the rise 
            while bond etfs have been on the decline so you may need to adjust the 
            allocation percentages accordingly depending on your risk tolerance.

    **Detailed Asset Class Breakdowns => using Vector Store**
    * Query the vector store to find information on relevant subcategories and investment options specific to that asset class
  and user's specific investment goals, investment horizon, and risk_tolerance. 
  * Analyze the retrieved data using historical performance, risk profiles, etc  based on the asset class type.
  * Based on this analysis and user input, recommend specific allocations for subcategories within the asset class. 
  * Explain the rationale behind the allocation percentages for each subcategory.""")
        
        return create_react_agent(llm, tools, state_modifier=system_prompt)


def create_vectordb():
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

def create_websearch_tool():
    web_search_tool = TavilySearchResults(max_results = 4)
    web_search_tool.description = """find relevant information and/or news about each specific asset 
    class in the user's investment portfolio from the internet to advise the user on 
    constructing their investment portfolio."""
    return web_search_tool


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
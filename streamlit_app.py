import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv(".env")
LLM_API_KEY = os.getenv("GEMINI_API_KEY")

st.title("🦜🔗 Wealthy Waldo: Your Investment Planning Assistant")

prompt_template = PromptTemplate.from_template(
    """You are Wealthy Waldo. You are an investment planning assistant who generates a personalized and specific 
    investment portfolio for a user based on the characteristics of their profile. Given a user with a {risk_tolerance}
    risk tolerance, {investment_goal} investment goal, and a {investment_horizon} investment horizon,  
    and considering the current market data and respective news for specific asset classes that you feel are necessary,  
    generate a diversified investment portfolio that aligns with the user's preferences.  
    Prioritize assets with {investment_style} investment style characteristics."""
)

def generate_response(input_text):
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.7, google_api_key=LLM_API_KEY)
    result = llm.invoke(input_text)
    st.info(result.content)

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
        generate_response(prompt_template.format(risk_tolerance = risk_tolerance_option, 
                                                investment_goal = investment_goals, 
                                                investment_horizon = investment_horizon_option, 
                                                investment_style = investment_styles_option))

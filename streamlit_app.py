import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(".env")
LLM_API_KEY = os.getenv("GEMINI_API_KEY")

st.title("🦜🔗 My Investment Guru")

def generate_response(input_text):
    llm = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.7, google_api_key=LLM_API_KEY)
    st.write(llm(input_text))

with st.form('my_form'):
    text = st.text_area('What can I help you with today?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)

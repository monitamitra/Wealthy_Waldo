# Wealthy Waldo: A GenAI-Powered Investment Planning Assistant
- Wealthy Waldo is an AI-powered investment planning assistant that generates personalized portfolio recommendations based on a user's risk tolerance, investment goal, and time horizon. Built with LangChain, OpenAI, and Streamlit, the app combines LLM reasoning with real-time financial data and user-uploaded context to create grounded, explainable investment strategies.

**App Link:** https://wealthy-waldo.streamlit.app/
---

## Key Features
- Personalized Recommendations: Users input their risk level, investment goals, and time horizon — and optionally upload personal financial notes to inform Waldo’s planning.
- Tool-Augmented LLM Agent: A custom LangChain ReAct agent coordinates multiple tools to enrich responses with real data and context.
- Hybrid RAG Pipeline: Combines a static text-based financial knowledge base (vectorized with FAISS) and user-uploaded Markdown/notes for personalized retrieval.
- Real-Time Market Insights: Integrates yFinance to fetch live market data (e.g., prices and daily % changes) for assets like VTI, BND, QQQ, and GLD.
- Web Search Integration: Uses Tavily Search to surface up-to-date financial news and trends, enhancing recommendations with current market conditions.
- Explainable Outputs: Responses include rationale for each asset allocation, supported by retrieved documents and live performance metrics.
---

## Technologies Used
- Frontend: Streamlit
- LLM: OpenAI (via LangChain)
- = RAG Stack: FAISS vector store + user-uploaded text files
- Tools: LangChain ReAct agent, Tavily Search, yFinance API
- Language: Python
---

## Screenshots

### User Input
![image](https://github.com/user-attachments/assets/e98c1ddc-7489-44fb-b945-291631e2d7e4)


### Agent Output
![image](https://github.com/user-attachments/assets/214a0182-dc16-41dd-97f5-745ef67f69f4)



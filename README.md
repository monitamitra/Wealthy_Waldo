# Wealthy Waldo: A GenAI-Powered Investment Planning Assistant
- Wealthy Waldo is an AI-powered investment planning assistant that generates personalized portfolio recommendations based on a user's risk tolerance, investment goal, time horizon, and a user's uploaded financial notes. 

**App Link:** https://wealthy-waldo-979073406187.us-central1.run.app/
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
- Frontend: Streamlit (Deployed on Google Cloud Run)
- LLM: OpenAI (via LangChain)
- = RAG Stack: FAISS vector store + user-uploaded text files
- Tools: LangChain ReAct agent, Tavily Search, yFinance API
- Language: Python
---

## Screenshots

### User Input
![image](https://github.com/user-attachments/assets/217a4977-35d2-45c2-aef8-82250efff619)

### Agent Output
![image](https://github.com/user-attachments/assets/69eaf370-3a1b-4cb9-b51b-faca01d26024)
![image](https://github.com/user-attachments/assets/be7c46e5-68b4-4edd-9f22-3f637ee7ed60)
![image](https://github.com/user-attachments/assets/3d119fea-ee8d-41c7-887b-9875a64a9e13)







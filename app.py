import streamlit as st
from rag_engine import semantic_search, generate_answer

st.title("📈 NIFTY 50 Stock RAG Chatbot")

query = st.text_input("Ask about stocks:")

if st.button("Analyze"):
    if query:
        results = semantic_search(query)
        st.write("Retrieved:", results)  # Keep temporarily

        answer = generate_answer(results, query)
        st.write("Model Output:", answer)
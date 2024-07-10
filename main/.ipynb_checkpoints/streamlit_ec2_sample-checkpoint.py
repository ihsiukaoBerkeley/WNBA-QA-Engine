import streamlit as st
import pandas as pd
import datetime

st.title("WNBA Data Chatbot")
st.write("Ask a question about the WNBA data:")

if 'responses' not in st.session_state:
    st.session_state.responses = []

def handle_query():
    query = st.session_state.query
    if query:
        response = f"Your input: '{query}'\n\nMy response: Come back next week for more answers :)"
        st.session_state.responses.append(response)
        st.session_state.query = ""

st.text_input("Your question:", key='query', on_change=handle_query)

for response in st.session_state.responses:
    st.write(response)

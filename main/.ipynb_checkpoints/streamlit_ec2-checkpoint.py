import streamlit as st
import pandas as pd
import datetime
from wnba_langchain import get_response

st.title("HoopsIQ - WNBA")
st.write("Ask a question about WNBA data:")
st.markdown(
    """
    <style>
    .big-divider {
        height: 5px;
        background-color: #333;
        margin: 20px 0;
    }
    </style>
    <div class="big-divider"></div>
    """,
    unsafe_allow_html=True
)

if 'responses' not in st.session_state:
    st.session_state.responses = []

def handle_query():
    query = st.session_state.query
    if query:
        langchain_response = get_response(query)
        streamlit_response = f"<b>Question:</b>\n\n{query}\n\n\n<b>Response:</b>\n\n{langchain_response}"
        st.session_state.responses.append(streamlit_response)
        st.session_state.query = ""
		
st.text_input("Your question:", key='query', on_change=handle_query)

for response in reversed(st.session_state.responses):
    st.markdown("---")
    st.markdown(response, unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import datetime
from wnba_langchain import get_response

# Define navigation function
def navigate_to(page):
    st.session_state.current_page = page
    
# Define page content
def home():
	st.title("HoopsIQ - WNBA")
	st.write("Ask a question about WNBA data:")
	st.markdown(
		"""
		<style>
		.big-divider {
			height: 10px;
			background-color: #333;
			margin: 10px 0;
		}
		</style>
		<div class="big-divider"></div>
		""",
		unsafe_allow_html=True
	)
	
	if 'questions' not in st.session_state:
		st.session_state.questions = []
	if 'visualizations' not in st.session_state:
		st.session_state.visualizations = []
	if 'responses' not in st.session_state:
		st.session_state.responses = []
	
	def handle_query():
		query = st.session_state.query
		if query:
			langchain_response, langchain_viz = get_response(query)
			st.session_state.questions.append(f"## Question:\n\n{query}\n\n\n")
			st.session_state.visualizations.append(langchain_viz)
			
			response_header = "Response:"
			if langchain_viz:
				response_header = "Further Response:"
			st.session_state.responses.append(f"## {response_header}\n\n{langchain_response}")
			
			st.session_state.query = ""
			
	st.text_input("Your question:", key='query', on_change=handle_query)

	for question, viz, response in zip(reversed(st.session_state.questions), reversed(st.session_state.visualizations), reversed(st.session_state.responses)):
		st.markdown(
			"""
			<style>
			.small-divider {
				height: 5px;
				background-color: #333;
				margin: 20px 0;
			}
			</style>
			<div class="small-divider"></div>
			""",
			unsafe_allow_html=True
		)
		st.markdown(question, unsafe_allow_html=True)
		if viz:
			st.markdown("## Visualization:\n\n", unsafe_allow_html=True)
			st.pyplot(viz)
		st.markdown(response, unsafe_allow_html=True)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'Home'

# Navigation links
st.sidebar.title("Navigation")
st.sidebar.button("Home", on_click=lambda: navigate_to("Home"))

# Render the appropriate page content
if st.session_state.current_page == "Home":
    home()

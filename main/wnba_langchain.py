import boto3
import json
import os
import requests
from langchain.agents import AgentExecutor
from langchain.agents import tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.chat_message_histories import (
    DynamoDBChatMessageHistory,
)
from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.globals import set_debug
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from urllib.parse import quote_plus

#

os.environ["OPENAI_API_KEY"] = "openai_api_key"

#

dynamodb = boto3.resource("dynamodb")

composite_key = {
    "SessionId": "session_id::110",
    "UserID": "0001",
}

chat_history = DynamoDBChatMessageHistory(
    table_name="Chat_Table",
    session_id="0",
    key = composite_key,
    history_size = 6,
)

#

AWS_REGION = "us-east-1"
SCHEMA_NAME = "wnba_db"
S3_STAGING_DIR = "s3://wnbadata/"

connect_str = "awsathena+rest://athena.{region_name}.amazonaws.com:443/{schema_name}?s3_staging_dir={s3_staging_dir}"

engine = create_engine(connect_str.format(
        region_name=AWS_REGION,
        schema_name=SCHEMA_NAME,
        s3_staging_dir=quote_plus(S3_STAGING_DIR)
))

db = SQLDatabase(engine)
schema = db.get_table_info()

#

f = open("src/query_example.json")
query_examples_json = json.load(f)
examples = query_examples_json["query_examples"]

#

file_paths = ["src/wnba_nba_pbp_data_dict.json", \
              "src/wnba_player_box.json", \
              "src/wnba_player_info.json", \
              "src/wnba_schedule.json", \
              "src/wnba_teambox.json"]

#

def get_table_details():
    table_details = ""
    for file_path in file_paths: 
        #load table names and descriptions from json
        f = open(file_path)
        table_dict = json.load(f)
        #retrieve table names and descriptions and compile into string
        table_details = table_details + "Table Name:" + table_dict['table_name'] + "\n" \
        + "Table Description:" + table_dict['table_description']
        for col in table_dict['values']:
            table_details = table_details + "\n" + "Column Name:" + col['column_name'] + "\n" \
            + "Column Description:" + col['column_description'] + "\n" \
            + "COlumn Type:" + col['column_type']
        table_details = table_details + "\n\n"
    return table_details
    
#

class Table(BaseModel):
    """Table in SQL database."""
    name: str = Field(description = "Name of table in SQL database.")

table_details = get_table_details()

table_prompt_system = f"""Refer the Above Context and Return the names of SQL Tables that MIGHT be relevant to the above context\n\n
The tables are:

{table_details}
"""

table_details_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", table_prompt_system),
        ("human", "{input}"),
    ]
)

table_chain_llm = ChatOpenAI(model_name= "gpt-3.5-turbo-0125", temperature=0)
table_chain_llm_wtools = table_chain_llm.bind_tools([Table])
output_parser = PydanticToolsParser(tools=[Table])

table_chain = table_details_prompt | table_chain_llm_wtools | output_parser

#

llm = ChatOpenAI(model_name= "gpt-3.5-turbo-0125", temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
llm_with_tools = llm.bind_tools(tools)

example_prompt = ChatPromptTemplate.from_messages(["User input: {input}\nSQL query: {query}"])
few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples = examples,
    example_prompt = example_prompt,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an agent designed to interact with a SQL database to answer questions about the WNBA.
        Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
        You should also leverage your pre-existing knowledge of WNBA rules, statistics, teams, players, and history to understand and interpret user questions and your answer accurately.
        Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
        You can order the results by a relevant column to return the most interesting examples in the database.
        Never query for all the columns from a specific table, only ask for the relevant columns given the question.
        You have access to tools for interacting with the database.
        Only use the below tools. Only use the information returned by the below tools to construct your final answer.
        Base your final answer solely on the information returned by these tools, combined with your existing knowledge of the WNBA.
        You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.
        
        DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
        
        To start you should ALWAYS look at the tables in the database to see what you can query.
        Do NOT skip this step.
        Then you should query the schema of the most relevant tables. Here is the relevant table info: {table_names_to_use}.""",
    ),
    few_shot_prompt,
    MessagesPlaceholder(variable_name = "chat_history"),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name = "agent_scratchpad"),
])

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
        "table_names_to_use": lambda x: x["table_names_to_use"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent = agent, tools = tools, verbose = False, return_intermediate_steps = True)

#

def get_response(input_text):
	if input_text:
		response = agent_executor.invoke({
			'input': input_text,
			'chat_history': chat_history.messages,
			'table_names_to_use': table_chain.invoke(input_text),
		})
		chat_history.add_user_message(input_text)
		chat_history.add_ai_message(response["output"])
		return response['output']

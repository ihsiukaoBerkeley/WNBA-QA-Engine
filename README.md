## WNBA QA Engine

### Description:
A RAG model using LangChain that asks ChatGPT to generate SQL query to fetch information stored in AWS S3 through AWS Athena based on user inputs.
The model would then generate context-aware responses based on user inputs and information stored in the database.

### Prototype (Streamlit UI hosted on AWS EC2: http://wnba.hoops-iq.com/):
<img width="477" alt="Screenshot 2024-07-10 at 7 52 16 PM" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/41d4148d-05a8-4541-9fe5-9d69c5180b6a">



### Tools:
- LangChain
- LLM models (e.g., ChatGPT)
- Streamlit
- AWS Athena
- AWS S3
- AWS EC2
- AWS DynamoDB

### Workflow Diagram:
#### Current Version (Text-to-SQL Chain with Query Examples and Table Descriptions, in langchain_text2sql_wExample_wTableDesc.ipynb):
<img width="741" alt="text2sql_wExample_wTableInfo" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/64df79dc-faf1-4f8c-8372-c571db0abe15">

#### V3 (Text-to-SQL Chain with Query Examples, Table Descriptions, and Table Context, in langchain_text2sql_wExample_wTableDesc_wContext.ipynb):
<img width="823" alt="text2sql_wExample_wTableInfo_wContext" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/a0413c48-1759-4b65-b331-4040491ce1a9">


#### V2 (Text-to-SQL Chain with Query Examples, in langchain_text2sql_wExample.ipynb):
<img width="741" alt="text2sql_wExample" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/06f9661a-ef8d-4b97-bea2-50ebfa9ecb94">

#### V1 (Text-to-SQL Chain, in langchain_text2sql.ipynb):
<img width="746" alt="text2sql_wHistory" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/2e52649f-821c-4ade-87b7-70f75265f16f">

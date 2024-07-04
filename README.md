## WNBA QA Engine

### Description:
A RAG model using LangChain that asks ChatGPT to generate SQL query to fetch information stored in AWS S3 through AWS Athena based on user inputs.
The model would then generate context-aware responses based on user inputs and information stored in the database.

### Prototype (Streamlit UI hosted on AWS EC2):
![image (2)](https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/f0380174-ef7d-47a5-b0c2-d2d562deba8e)


### Tools:
- LangChain
- LLM models (e.g., ChatGPT)
- Streamlit
- AWS Athena
- AWS S3
- AWS EC2
- AWS DynamoDB

### Workflow Diagram:
#### Current Version (Text-to-SQL Chain with Query Examples and Table Descriptions):
<img width="741" alt="text2sql_wExample_wTableInfo" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/64df79dc-faf1-4f8c-8372-c571db0abe15">

#### V2 (Text-to-SQL Chain with Query Examples):
<img width="741" alt="text2sql_wExample" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/06f9661a-ef8d-4b97-bea2-50ebfa9ecb94">

#### V1 (Text-to-SQL Chain):
<img width="746" alt="text2sql_wHistory" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/2e52649f-821c-4ade-87b7-70f75265f16f">

## WNBA QA Engine

### Description:
A text2sql RAG model using LangChain to fetch data stored in AWS S3 through Athena based on user inputs. 
The model would then generate context-aware responses based on user inputs and information retrieved from the database.
#### Problems to Address:
1. WNBA data is extremely difficult to extract for non-basketball data-savvy users.
2. Publicly available LLMs also consistently report false information.

### Target Users:
Daily Fantasy Players: massive viewership growth in the WNBA over the last few seasons


### Prototype (Streamlit UI hosted on AWS EC2: http://wnba.hoops-iq.com/):
<img width="517" alt="Screenshot 2024-07-31 at 12 03 24 AM" src="https://github.com/user-attachments/assets/943270a4-5e73-406b-9384-b056c7f13afe">

### Tools:
- LangChain
- LLM models (e.g., ChatGPT)
- Streamlit
- AWS Athena
- AWS S3
- AWS EC2
- AWS DynamoDB

#### SOTA LLMs Evalution:
(Number of Questions Answered Correctly - Hallucinations) / Total Number of Questions (=38)

| Model  | Baseline | QA Engine | 
| -------| ------- | --------- | 
| GPT 3.5 Turbo | -58% (-25) | 16% (-13) |
| GPT 4 Turbo | -11% (-12) | 45% (-9) |
| GPT 4o | -24% (-18) | 47% (-9) |
| GPT 4o mini | -32% (-21) | 42% (-9) |
| Claude 3.5 Sonnet | 0% (-8) | 45% (-10) |

### Workflow Diagram:
#### Current Version (V5) (Text-to-SQL Chain with Query Examples and Table Descriptions with Multi-Modality, in langchain_text2sql_wExample_wTableDesc_MultiModal.ipynb):
<img width="800" alt="Screenshot 2024-07-30 at 11 36 45 PM" src="https://github.com/user-attachments/assets/6386bd8a-d928-478b-9a20-ea7bce8e7eec">

#### V6 (Text-to-SQL Chain with Query Examples and Table Descriptions, and Term Context in langchain_text2sql_wExample_wTableDesc_wTermContext.ipynb):
<img width="800" alt="Screenshot 2024-07-30 at 11 40 03 PM" src="https://github.com/user-attachments/assets/a0a9d700-0988-405d-8ded-3a1d13c1fa9d">

#### V4 (Text-to-SQL Chain with Query Examples, Table Descriptions, and Table Context, in langchain_text2sql_wExample_wTableDesc_wContext.ipynb):
<img width="823" alt="text2sql_wExample_wTableInfo_wContext" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/a0413c48-1759-4b65-b331-4040491ce1a9">

#### V3 (Text-to-SQL Chain with Query Examples and Table Descriptions, in langchain_text2sql_wExample_wTableDesc.ipynb):
<img width="741" alt="text2sql_wExample_wTableInfo" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/64df79dc-faf1-4f8c-8372-c571db0abe15">

#### V2 (Text-to-SQL Chain with Query Examples, in langchain_text2sql_wExample.ipynb):
<img width="741" alt="text2sql_wExample" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/06f9661a-ef8d-4b97-bea2-50ebfa9ecb94">

#### V1 (Text-to-SQL Chain, in langchain_text2sql.ipynb):
<img width="746" alt="text2sql_wHistory" src="https://github.com/ihsiukaoBerkeley/WNBA-QA-Engine/assets/117419224/2e52649f-821c-4ade-87b7-70f75265f16f">

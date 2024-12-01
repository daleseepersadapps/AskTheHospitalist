import os
import openai
from openai import OpenAI
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import streamlit as st
from langchain import OpenAI
import asyncio
from openai import AsyncOpenAI
import json

os.environ["OPENAI_API_KEY"] = st.secrets["APIKey_OpenAI"]
client = OpenAI()
client = AsyncOpenAI(api_key=st.secrets["APIKey_OpenAI"])

st.set_page_config(page_icon="💊", page_title="Ask The Hospitalist")
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

OPENAI_API_KEY = st.secrets["APIKey_OpenAI"]
PINECONE_API_KEY = st.secrets["APIKey_Pinecone"]
#PINECONE_API_ENV = 'us-east-1'
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
            </style>
            """

pc = Pinecone(api_key=st.secrets["APIKey_Pinecone"], GRPC_DNS_RESOLVER="native")
index = pc.Index("uptodate")

openai.api_key = st.secrets["APIKey_OpenAI"]

st.title("Ask the Hospitalist")
query = st.text_area("Ask me any Internal Medicine Board MCQ", height=300)
def handle_button_click():
    st.session_state.button_disabled = True
if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = False
col1, col2 = st.columns(2)
with col1:
    if st.button("Ask", on_click=handle_button_click, disabled=st.session_state.button_disabled):
        handle_button_click()
with col2:
    if st.button('Stop'):
        st.stop()
        st.session_state.clear()

res_box=st.empty()

model_engine = "gpt-4o"
embed_model = "text-embedding-ada-002"

primer_for_extracting_reasoning = ("""You are an expert internal medicine doctor. You will be presented with a complex problem. Your job is to formulate a plan of how you will go about solving this problem. Describe in detail how you will solve this problem. Say how you will use the information you requested to derive your answer. Think of it as making instructions for someone to follow that will guide them to the answer, but do not solve it. You are sort of making a plan or an algorirthm. 
You will be given a tool called "ENCYCLOPEDIA." ENCYCLOPEDIA is a tool where you can research medical knowledge by asking it a number of questions. Remember, **this encyclopedia is agnostic to the original question so formulate your questions appropriately**. In your plan, say when you will use this tool and say what questions you will ask it.  Also say **why each question is important** and **how you will use the information obtained to guide through the reasoning**. **DO NOT ANSWER THE QUESTION. YOUR JOB IS TO FORMULATE A PLAN.**
""")
primer_for_extracting_JSON = """Run through the program and create a list of ALL the questions that must be passed to the ENCYCLOPEDIA tool. Do not include a variable placeholder, instead say what is the word that should be there. If the question is part of a loop, include all possibilities in your list. For each question, include why is it important and how the information obtained is going to help solve the problem. Present your answer as a JSON list.
eg:
    {
        "question": "What are the recommended next steps if a patient does not respond to initial non-invasive management for dyspepsia?",
        "importance": "To plan for further management in case symptoms persist.",
        "usage": "To understand the progression of diagnostic evaluations, such as the eventual role of upper endoscopy."
    }
    """

async def main():
    if query.strip() :
        with st.spinner(text="Thinking... This can take up to 1 min. Please do not click ask again.",):
             #STEP 1: GENERATE REASONING
            response = await client.chat.completions.create(
                model="o1-preview",
                messages=[
                    {"role": "user", "content": primer_for_extracting_reasoning + "\nQuestion: " + query}
                ]
            )
            reasoning = response.choices[0].message.content.strip()  # Initial response with questions unstructured
        
            # STEP 2 EXTRACT JSON QUESTIONS
            response_json = await client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                messages=[
                    {"role": "system", "content": primer_for_extracting_JSON},
                    {"role": "user", "content": "Question: " + reasoning},
                ],
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "medical_question_plan",
                        "schema": {
                            "type": "object",
                            "properties": {
                            "questions": {
                                "type": "array",
                                "description": "A list of questions to be extracted from the medical plans.",
                                "items": {
                                "type": "object",
                                "properties": {
                                    "question": {
                                    "type": "string",
                                    "description": "The specific medical question being asked."
                                    },
                                    "importance": {
                                    "type": "string",
                                    "description": "Why this question is important in the medical context."
                                    },
                                    "usage": {
                                    "type": "string",
                                    "description": "How the information will be used or applied in the context of the medical plan."
                                    }
                                },
                                "required": [
                                    "question",
                                    "importance",
                                    "usage"
                                ],
                                "additionalProperties": False
                                }
                        }
                        },
                        "required": [
                            "questions"
                            ],
                        "strict": True,
                        "additionalProperties": False
                    }
                    }
                }
            )
            JSON_output = response_json.choices[0].message.content # JSON OUTPUT
            print (JSON_output)

            # STEP 3 Iterate over each question
            questions = json.loads(JSON_output)

            # Function to process each question asynchronously
            async def get_response(prompt, display_question):
                with st.spinner(text= display_question):
                    response_subquestion = await client.chat.completions.create(
                        model="gpt-4o",
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": """analyze the key concepts in the following question. Instead of answering it, generate the top 5 educational objective questions that can be researched to better answer the question. 
                            The educational objective questions must consider the context of the original question and include the key concepts."""},
                            {"role": "user", "content": prompt},  # prompt is each question
                        ]
                    )

                    steps_tasks = (response_subquestion.choices[0].message.content)

                    res = await client.embeddings.create(
                        input=steps_tasks,
                        model=embed_model,
                    )

                    xq = (res.data[0].embedding)
                    res = index.query(vector=xq, top_k=10, include_metadata=True)
                    contexts = [item['metadata']['text'] for item in res['matches']]
                    augmented_query = ("\n---\n".join(contexts)+"\n-----\n" + prompt)

                    primer = f"""You are Q&A bot talking to a doctor. A highly intelligent system that answers the doctor's questions based on the information provided by the user above each question. Be detailed in your answer. Draw a table when you are asked to. Use medical jargon wherever possible. Do not start your response with, 'based on the information' or a similar phrase. If the information cannot be found in the information provided by the user you truthfully say "I don't know". Always follow the information given and do not make our own assumptions."""
                    resp = await client.chat.completions.create(
                        model="gpt-4-turbo",
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": primer},
                            {"role": "user", "content": "based on the information above, " + augmented_query + " Quote which pieces of information you derived your answer from"}
                        ]
                    )
                    question_answer = resp.choices[0].message.content
                    return question_answer

            tasks = []
            for q in questions["questions"]:
                display_question = q['question']
                prompt = (
                    f"Question:\n{q['question']}\n\n"
                    f"Why it's important:\n{q['importance']}\n\n"
                    f"How it will be used:\n{q['usage']}\n\n"
                    "Please provide a detailed answer."
                )
                tasks.append(get_response(prompt, display_question))

            responses = await asyncio.gather(*tasks)
            concatenated_answers = "\n\n---\n\n".join(responses)
            print(concatenated_answers)

            finalanswer = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": "You will be presented with a complex medical question and a plan of how to reason through it. You will also be given factual medical knowledge that you must use to solve it. Analyze the information given and come to a conclusion. Question: " + query + "Reasoning plan: " + reasoning + "Medical facts from encyclopedia: " + concatenated_answers}
                ]
            )
            answer = finalanswer.choices[0].message.content.strip()
            st.write(answer)

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())

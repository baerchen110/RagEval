from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from openai import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
import os
import csv
import json
from dotenv import load_dotenv
from dotenv import dotenv_values
import random
from tqdm import tqdm
import pandas as pd
from IPython.display import display
import datasets
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import SystemMessage

# Load environment variables from .env file
if os.path.exists(".env"):
    load_dotenv(override=True)
    config = dotenv_values(".env")

API_BASE = os.getenv("AZURE_OPENAI_BASE")
API_KEY = os.getenv("AZURE_OPENAI_KEY")
API_TYPE = os.environ.get("AZURE_OPENAI_TYPE", "azure")
API_VERSION = os.getenv("AZURE_OPENAI_VERSION")
ENGINE = os.getenv("AZURE_OPENAI_DEPLOYMENT")
MODEL = os.getenv("AZURE_OPENAI_MODEL")
ANSWER_PATH = '../data/RAG_EVAL3_results.csv'


openai_client = AzureOpenAI(api_version=API_VERSION,
                            azure_endpoint=API_BASE,
                            api_key=API_KEY,
                            azure_deployment=ENGINE
                            )

llm_client = AzureChatOpenAI(
    openai_api_version=API_VERSION,
    azure_endpoint=API_BASE,
    openai_api_key=API_KEY,
    azure_deployment=ENGINE,
    temperature=1
)


EVALUATION_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: {{write a feedback for criteria}} [RESULT] {{an integer number between 1 and 5}}\"
4. Please do not generate any other opening, closing, and explanations. Be sure to include [RESULT] in your output.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference_answer}

###Score Rubrics:
[Is the response correct, accurate, and factual based on the reference answer?]
Score 1: The response is completely incorrect, inaccurate, and/or not factual.
Score 2: The response is mostly incorrect, inaccurate, and/or not factual.
Score 3: The response is somewhat correct, accurate, and/or factual.
Score 4: The response is mostly correct, accurate, and factual.
Score 5: The response is completely correct, accurate, and factual.

###Feedback:"""



evaluation_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content="You are a fair evaluator language model."),
        HumanMessagePromptTemplate.from_template(EVALUATION_PROMPT),
    ]
)


if __name__ == "__main__":
    if not os.path.exists(ANSWER_PATH):
        print(f"Error: File '{ANSWER_PATH}' not found.")

    try:
        with open(ANSWER_PATH, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)

            # Check if required columns exist
            required_columns = ['question', 'ref', 'vector', 'bm25', 'rrf']
            header = csv_reader.fieldnames

            if not all(col in header for col in required_columns):
                missing_cols = [col for col in required_columns if col not in header]
                print(f"Error: Missing columns in CSV file: {', '.join(missing_cols)}")

            total_scores_vector = []
            total_scores_bm25 = []
            total_scores_rrf = []

            # Process each row
            for row in csv_reader:
                question = row['question']
                ref = row['ref']
                vector = row['vector']
                bm25 = row['bm25']
                rrf = row['rrf']

                #vector
                eval_prompt = evaluation_prompt_template.format_messages(
                    instruction=question,
                    response=vector,
                    reference_answer=ref,
                )

                eval_result = llm_client.invoke(eval_prompt)

                feedback, score = [
                    item.strip() for item in eval_result.content.split("[RESULT]")
                ]

                total_scores_vector.append(int(score))

                #rrf
                eval_prompt = evaluation_prompt_template.format_messages(
                    instruction=question,
                    response=rrf,
                    reference_answer=ref,
                )

                eval_result = llm_client.invoke(eval_prompt)

                feedback, score = [
                    item.strip() for item in eval_result.content.split("[RESULT]")
                ]

                total_scores_rrf.append(int(score))

                #bm25
                eval_prompt = evaluation_prompt_template.format_messages(
                    instruction=question,
                    response=bm25,
                    reference_answer=ref,
                )

                eval_result = llm_client.invoke(eval_prompt)

                feedback, score = [
                    item.strip() for item in eval_result.content.split("[RESULT]")
                ]

                total_scores_bm25.append(int(score))


                print(f"Question: {question}")
                print(f"Ref: {ref}")
                print(f"Vector: {vector}")
                print(f"score: {score}")
                print(f"feedback: {feedback}")
                print("-" * 50)  # Print a separator between rows

            if total_scores_vector:
                average_score = sum(total_scores_vector) / len(total_scores_vector)
                print(f"\nAverage Score vector: {average_score:.2f}/5.00")
                print(f"Total Evaluated: {len(total_scores_vector)} questions")
            else:
                print("No valid scores were calculated")

            if total_scores_rrf:
                average_score = sum(total_scores_rrf) / len(total_scores_rrf)
                print(f"\nAverage Score rrf: {average_score:.2f}/5.00")
                print(f"Total Evaluated: {len(total_scores_rrf)} questions")
            else:
                print("No valid scores were calculated")

            if total_scores_bm25:
                average_score = sum(total_scores_bm25) / len(total_scores_bm25)
                print(f"\nAverage Score bm25: {average_score:.2f}/5.00")
                print(f"Total Evaluated: {len(total_scores_bm25)} questions")
            else:
                print("No valid scores were calculated")

    except Exception as e:
        print(f"Error processing file: {e}")

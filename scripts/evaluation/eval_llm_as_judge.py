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
if os.path.exists("../.env"):
    load_dotenv(override=True)
    config = dotenv_values("../.env")

API_BASE = os.getenv("AZURE_OPENAI_BASE")
API_KEY = os.getenv("AZURE_OPENAI_KEY")
API_TYPE = os.environ.get("AZURE_OPENAI_TYPE", "azure")
API_VERSION = os.getenv("AZURE_OPENAI_VERSION")
ENGINE = os.getenv("AZURE_OPENAI_DEPLOYMENT")
MODEL = os.getenv("AZURE_OPENAI_MODEL")
ANSWER_PATH = ["../data/fulltext_answers.json", "../data/hybrid_answers.json", "../data/vector_answers.json", "../data/rerank_answers.json"]
OUTPUT_FILE_EVALUATION = '../../data/eval/medical/multi/eval.json'


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

    output_file = OUTPUT_FILE_EVALUATION
    try:  # load previous generations if they exist
        with open(output_file, "r") as out_f:
            outputs = json.load(out_f)
    except:
        outputs = []

    for path in ANSWER_PATH:
        print(f"Loading: File '{path}'")

        if not os.path.exists(path):
            print(f"Error: File '{path}' not found.")

        try:
            with open(path, 'r') as file:
                data = json.load(file)
                total_scores = []
                for doc in data:

                    question = doc.get('question')
                    context = doc.get('retrieved_context')
                    ref_anwser = doc.get('ref_answer')
                    generated_answer = doc.get('generated_answer')

                    eval_prompt = evaluation_prompt_template.format_messages(
                        instruction=question,
                        response=generated_answer,
                        reference_answer=ref_anwser,
                    )

                    eval_result = llm_client.invoke(eval_prompt)

                    feedback, score = [
                        item.strip() for item in eval_result.content.split("[RESULT]")
                    ]

                    total_scores.append(int(score))

                    print(f"Question: {question}")
                    print(f"Ref: {ref_anwser}")
                    print(f"Answer: {generated_answer}")
                    print(f"score: {score}")
                    print(f"feedback: {feedback}")
                    print("-" * 50)  # Print a separator between rows

                if total_scores:
                    average_score = sum(total_scores) / len(total_scores)
                    print(f"\nAverage Score: {average_score:.2f}/5.00")
                    print(f"Total Evaluated: {len(total_scores)} questions")
                    print(f"\nfile: {path}")

                    result = {
                        "average_score": average_score,
                        "total_questions": len(total_scores),
                        "path": path,
                        "total_scores": total_scores
                    }
                    outputs.append(result)

                    with open(output_file, "w") as f:
                        json.dump(outputs, f)
                else:
                    print("No valid scores were calculated")


        except Exception as e:
            print(f"Error processing file: {e}")

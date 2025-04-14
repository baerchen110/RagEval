## Install the required packages
## pip install -qU elasticsearch openai
from elasticsearch import Elasticsearch
from openai import OpenAI
from dotenv import load_dotenv
from dotenv import dotenv_values
import os
from openai import AzureOpenAI
import pandas as pd


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
OUTPUT_FILE_QUESTIONS = '../data/RAG_Eval3_sample_medical_en.csv'

es_client = Elasticsearch(
    cloud_id=os.environ["ES_CID"],
    basic_auth=(os.environ["ES_USER"], os.environ["ES_PWD"])
)

openai_client = AzureOpenAI(api_version=API_VERSION,
                            azure_endpoint=API_BASE,
                            api_key=API_KEY,
                            azure_deployment=ENGINE
                            )


index_source_fields = {
    "eval-rag-medical-en-1": [
        "content_semantic"
    ]
}

def get_elasticsearch_results(query:str):
    es_query = {
        "retriever": {
            "standard": {
                "query": {
                    "nested": {
                        "path": "content_semantic.inference.chunks",
                        "query": {
                            "sparse_vector": {
                                "inference_id": "my-elser-endpoint",
                                "field": "content_semantic.inference.chunks.embeddings",
                                "query": query
                            }
                        },
                        "inner_hits": {
                            "size": 2,
                            "name": "eval-rag-medical-en-1.content_semantic",
                            "_source": [
                                "content_semantic.inference.chunks.text"
                            ]
                        }
                    }
                }
            }
        },
        "size": 1
    }
    result = es_client.search(index="eval-rag-medical-en-1", body=es_query)
    return result["hits"]["hits"]


def create_openai_prompt(results):
    context = ""
    for hit in results:
        inner_hit_path = f"{hit['_index']}.{index_source_fields.get(hit['_index'])[0]}"
        ## For semantic_text matches, we need to extract the text from the inner_hits
        if 'inner_hits' in hit and inner_hit_path in hit['inner_hits']:
            context += '\n --- \n'.join(
                inner_hit['_source']['text'] for inner_hit in hit['inner_hits'][inner_hit_path]['hits']['hits'])
        else:
            source_field = index_source_fields.get(hit["_index"])[0]
            hit_context = hit["_source"][source_field]
            context += f"{hit_context}\n"
    prompt = f"""
    <|system|>
    Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked, response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.</s>
    <|user|>

  Context:
  {context}

  """
    return prompt


def generate_openai_completion(user_prompt, question):
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": user_prompt},
            {"role": "user", "content": question},
        ],
        temperature=1
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    file_name = OUTPUT_FILE_QUESTIONS
    data = pd.read_csv(file_name)
    # Extract the 'question' column
    questions = data['question'].tolist()
    for question in questions:
        elasticsearch_results = get_elasticsearch_results(question)
        context_prompt = create_openai_prompt(elasticsearch_results)
        openai_completion = generate_openai_completion(context_prompt, question)
        print(f"\n**Question :**")
        print(question)
        print("**Answer:**")
        print(openai_completion)
        print("-" * 50)

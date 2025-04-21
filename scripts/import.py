from langchain_community.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
from openai import AzureOpenAI
import os
import json
from dotenv import load_dotenv
from dotenv import dotenv_values
import random
from tqdm import tqdm
import pandas as pd
from IPython.display import display
import datasets
import json

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
RAW_DATA_FILE = "../data/dragonball_docs_med_en_1.json"
OUTPUT_GROUND_TRUTH_FILE = "../data/med_ground_truth.json"

loader = JSONLoader(
    file_path=RAW_DATA_FILE,
    jq_schema=".[].content",
    text_content=True
)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    add_start_index=True,
    separators=["\n\n", "\n", ".", " ", ""],
)


docs_processed = []

for doc in documents:
    #print(f"Document Content:\n{doc.page_content},...\n")
    docs_processed += text_splitter.split_documents([doc])
    #print(f"Metadata: {doc.metadata},\n{'-'*50},")

for doc in docs_processed:
    print(f"Docs: {doc}")

client = AzureOpenAI(api_version=API_VERSION,
                     azure_endpoint=API_BASE,
                     api_key=API_KEY,
                     azure_deployment=ENGINE
                     )


def call_llm(client: AzureOpenAI, prompt: str):
    response = client.chat.completions.create(
        model=ENGINE,  # or use the model name if different
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000
    )
    return response.choices[0].message.content

# Test the call
print(call_llm(client, "This is a test context, what is your name"))

QA_generation_prompt = """
Your task is to write a factoid question and an answer given a context.
Your factoid question should be answerable with a specific, concise piece of factual information from the context.
Your factoid question should be formulated in the same style as questions users could ask in a search engine.
This means that your factoid question MUST NOT mention something like "according to the passage" or "context".

Provide your answer as follows:

Output:::
Factoid question: (your factoid question)
Answer: (your answer to the factoid question)

Now here is the context.

Context: {context}\n
Output:::"""


N_GENERATIONS = min(600, len(docs_processed))   # We intentionally generate only 10 QA couples here for cost and time considerations

print(f"Generating {N_GENERATIONS} QA couples...")

outputs = []
for sampled_context in tqdm(random.sample(docs_processed, N_GENERATIONS), desc="Generating QAs"):
    try:
        output_QA_couple = call_llm(
            client,
            QA_generation_prompt.format(context=sampled_context.page_content)
        )

        # More robust parsing
        if "Factoid question:" in output_QA_couple and "Answer:" in output_QA_couple:
            parts = output_QA_couple.split("Factoid question:")[-1].split("Answer:")
            question = parts[0].strip()
            answer = parts[1].strip()

            if len(answer) >= 300:
                continue  # Skip too long answers

            outputs.append({
                "context": sampled_context.page_content,
                "question": question,
                "answer": answer,
                "source_doc": sampled_context.metadata.get("source", "unknown"),
            })
    except Exception as e:
        print(f"Error processing document: {e}")
        continue

# Display results
df = pd.DataFrame(outputs)

# Configure Pandas to show ALL data without truncation
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Adjust based on your console width
pd.set_option('display.max_colwidth', 50)  # Adjust for text columns

# Print the full DataFrame
print(df.to_string())  # to_string() ensures proper formatting


question_groundedness_critique_prompt = """
You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here are the question and context.

Question: {question}\n
Context: {context}\n
Answer::: """

question_relevance_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how useful this question can be to doctors, medical school students or medical researchers.
Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """

question_standalone_critique_prompt = """
You will be given a question.
Your task is to provide a 'total rating' representing how context-independent this question is.
Give your answer on a scale of 1 to 5, 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself
Provide your answer as follows:

Answer:::
Evaluation: (your rationale for the rating, as a text)
Total rating: (your rating, as a number between 1 and 5)

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Now here is the question.

Question: {question}\n
Answer::: """


print("Generating critique for each QA couple...")
for output in tqdm(outputs):
    evaluations = {
        "groundedness": call_llm(
            client,
            question_groundedness_critique_prompt.format(
                context=output["context"], question=output["question"]
            ),
        ),
        "relevance": call_llm(
            client,
            question_relevance_critique_prompt.format(question=output["question"]),
        ),
        "standalone": call_llm(
            client,
            question_standalone_critique_prompt.format(question=output["question"]),
        ),
    }
    try:
        for criterion, evaluation in evaluations.items():
            score, eval = (
                int(evaluation.split("Total rating: ")[-1].strip()),
                evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
            )
            output.update(
                {
                    f"{criterion}_score": score,
                    f"{criterion}_eval": eval,
                }
            )
    except Exception as e:
        continue

pd.set_option("display.max_colwidth", None)

generated_questions = pd.DataFrame.from_dict(outputs)

print("Evaluation dataset before filtering:")
display(
    generated_questions[
        [
            "question",
            "answer",
            "groundedness_score",
            "relevance_score",
            "standalone_score",
        ]
    ]
)
generated_questions = generated_questions.loc[
    (generated_questions["groundedness_score"] >= 4)
    & (generated_questions["relevance_score"] >= 3)
    & (generated_questions["standalone_score"] >= 3)
]
print("============================================")
print("Final evaluation dataset:")
display(
    generated_questions[
        [
            "question",
            "answer",
            "context",
            "groundedness_score",
            "relevance_score",
            "standalone_score",
        ]
    ]
)

output_file = OUTPUT_GROUND_TRUTH_FILE
try:  # load previous generations if they exist
    with open(output_file, "r") as f:
        outputs = json.load(f)
except:
    outputs = []

# Convert DataFrame to dictionary with records orientation
generated_questions_dict = generated_questions.to_dict(orient='records')
outputs.append(generated_questions_dict)

with open(output_file, "w") as f:
    json.dump(outputs, f)

eval_dataset = datasets.Dataset.from_pandas(
    generated_questions, split="train", preserve_index=False
)

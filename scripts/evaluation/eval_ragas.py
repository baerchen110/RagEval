from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
import os
import json
from dotenv import load_dotenv
from dotenv import dotenv_values

from datasets import Dataset
from ragas import evaluate,RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, context_precision, LLMContextPrecisionWithReference
from ragas.metrics import LLMContextRecall


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
ANSWER_PATH = ["../data/law_en_fulltext_answers.json","../data/law_en_bge_answers.json", "../data/law_en_hybrid_bge_answers.json", "../data/law_en_hybrid_linear_bge_answers.json", "../data/law_en_hybrid_rerank_bge_answers.json"]
OUTPUT_FILE_EVALUATION = '../../data/eval/medical/multi/eval_ragas.json'

API_VERSION_EMBEDDING = os.getenv("AZURE_API_VERSION_EMBEDDING")
ENGINE_EMBEDDING = os.getenv("AZURE_ENGINE_EMBEDDING")
API_KEY_EMBEDDING = os.getenv("AZURE_API_KEY_EMBEDDING");

llm_client = AzureChatOpenAI(
    openai_api_version=API_VERSION,
    azure_endpoint=API_BASE,
    openai_api_key=API_KEY,
    azure_deployment=ENGINE,
    temperature=1
)

azure_embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=API_BASE,
    api_key=API_KEY_EMBEDDING,
    api_version=API_VERSION_EMBEDDING,
    azure_deployment="text-embedding-3-large",
    model="text-embedding-3-large"
)


if __name__ == "__main__":
    # Wrap models for Ragas compatibility
    ragas_llm = LangchainLLMWrapper(llm_client)
    ragas_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)

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
                total_scores_faithfulness = []
                total_scores_answer_relevancy = []
                total_scores_context_precision = []
                total_scores_context_recall = []


                for doc in data:
                    question = doc.get('question')
                    contexts = doc.get('raw_context')
                    ref_anwser = doc.get('ref_answer')
                    generated_answer = doc.get('generated_answer')

                    dataset_dict = {
                        "question": [question],
                        "answer": [generated_answer],
                        "contexts": [contexts],
                        "ground_truth": [
                            ref_anwser]
                    }

                    run_config = RunConfig(timeout=120)
                    dataset = Dataset.from_dict(dataset_dict)
                    metrics = [
                        Faithfulness(llm=ragas_llm),
                        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
                        LLMContextPrecisionWithReference(llm=ragas_llm),  # Measures if relevant items in context are ranked higher[2]
                        LLMContextRecall(llm=ragas_llm)
                    ]

                    # Run evaluation
                    result = evaluate(
                        dataset=dataset,
                        metrics=metrics,
                        run_config=run_config
                    )

                    # Convert to pandas and save results
                    df = result.to_pandas()
                    df.to_csv("../data/rag_evaluation_results.csv", index=False)


                    print("Evaluation Results:")
                    # Proper score handling
                    if isinstance(result['faithfulness'], list):
                        avg_score = sum(result['faithfulness']) / len(result['faithfulness'])
                        total_scores_faithfulness.extend(result['faithfulness'])
                        print(f"Faithfulness: {avg_score:.2f}")
                    else:  # Handle single-sample edge case
                        print(f"Faithfulness: {result['faithfulness']:.2f}")

                    # Proper score handling
                    if isinstance(result['answer_relevancy'], list):
                        avg_score = sum(result['answer_relevancy']) / len(result['answer_relevancy'])
                        print(f"Answer Relevancy: {avg_score:.2f}")
                        total_scores_answer_relevancy.extend(result['answer_relevancy'])
                    else:  # Handle single-sample edge case
                        print(f"Answer Relevancy: {result['answer_relevancy']:.2f}")
                    #print(f"Context Precision: {result['context_precision']:.2f}")
                    #print(f"Context Recall: {result['context_recall']:.2f}")


                    if isinstance(result['llm_context_precision_with_reference'], list):
                        avg_score = sum(result['llm_context_precision_with_reference']) / len(result['llm_context_precision_with_reference'])
                        print(f"Context precision: {avg_score:.2f}")
                        total_scores_context_precision.extend(result['llm_context_precision_with_reference'])
                    else:  # Handle single-sample edge case
                        print(f"Context precision: {result['llm_context_precision_with_reference']:.2f}")

                    if isinstance(result['context_recall'], list):
                        avg_score = sum(result['context_recall']) / len(result['context_recall'])
                        print(f"Context recall: {avg_score:.2f}")
                        total_scores_context_recall.extend(result['context_recall'])
                    else:  # Handle single-sample edge case
                        print(f"Context recall: {result['context_recall']:.2f}")



                if total_scores_faithfulness:
                    print(f"\nfile: {path}")
                    print(f"Total Evaluated: {len(total_scores_faithfulness)} questions")
                    average_score_faithfulness = sum(total_scores_faithfulness) / len(total_scores_faithfulness)
                    print(f"\nAverage Score Faithfulness: {average_score_faithfulness:.2f}")

                    average_score_answer_relevancy = sum(total_scores_answer_relevancy) / len(total_scores_faithfulness)
                    print(f"\nAverage Score answer relevancy: {average_score_answer_relevancy:.2f}")


                    average_score_context_precision = sum(total_scores_context_precision) / len(total_scores_context_precision)
                    print(f"\nAverage Score context precision: {average_score_context_precision:.2f}")


                    average_score_context_recall = sum(total_scores_context_recall) / len(total_scores_context_recall)
                    print(f"\nAverage Score context recall: {average_score_context_recall:.2f}")



                    result = {
                        "average_score_faithfulness": average_score_faithfulness,
                        "average_score_answer_relevancy": average_score_answer_relevancy,
                        "average_score_context_precision": average_score_context_precision,
                        "average_score_context_recall": average_score_context_recall,
                        "total_questions": len(total_scores_faithfulness),
                        "path": path,
                        "total_scores_faithfulness": total_scores_faithfulness,
                        "total_scores_answer_relevancy": total_scores_answer_relevancy,
                        "total_scores_context_precision": total_scores_context_precision,
                        "total_scores_context_recall": total_scores_context_recall
                    }
                    outputs.append(result)

                    with open(output_file, "w") as f:
                        json.dump(outputs, f)
                else:
                    print("No valid scores were calculated")


        except Exception as e:
            print(f"Error processing file: {e}")

import os
import time
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.eval_data import TEST_CASES
from src.pipeline_run import build_pipeline, run_query

load_dotenv()

# Configure RAGAS to use Gemini as both the judge LLM and the embedding model
judge_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.environ["GEMINI_API_KEY"])
judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.environ["GEMINI_API_KEY"])


def run_config(use_semantic_chunking: bool, use_reranking: bool):
    """
    Runs all test questions through one specific pipeline configuration,
    then scores the results with RAGAS.
    """
    print(f"\n=== Config: semantic_chunking={use_semantic_chunking}, reranking={use_reranking} ===")

    store, hybrid = build_pipeline(use_semantic_chunking)

    questions, contexts_list, answers, references = [], [], [], []

    for case in TEST_CASES:
        question = case["question"]
        reference = case["reference"]

        retrieved_contexts, generated_answer = run_query(question, store, hybrid, use_reranking)

        questions.append(question)
        contexts_list.append(retrieved_contexts)
        answers.append(generated_answer)
        references.append(reference)

        time.sleep(2)

    dataset = Dataset.from_dict({
        "question": questions,
        "contexts": contexts_list,
        "answer": answers,
        "reference": references,
    })

    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    return result


if __name__ == "__main__":
    configs = [
        {"use_semantic_chunking": False, "use_reranking": False},
        {"use_semantic_chunking": False, "use_reranking": True},
        {"use_semantic_chunking": True, "use_reranking": False},
        {"use_semantic_chunking": True, "use_reranking": True},
    ]

    for config in configs:
        try:
            result = run_config(**config)
            print(result)
        except Exception as e:
            print(f"Config{config} Failed :{e}")
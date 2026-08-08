import argparse
import os
from datetime import datetime

from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from src.eval_data import TEST_CASES
from src.pipeline_run import build_pipeline, run_query

load_dotenv()

# Keep at 1 until a config runs clean end to end; raise to ~4 after.
MAX_WORKERS = 1
RESULTS_DIR = "results"

judge_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    vertexai=True,
    project=os.environ["VERTEX_PROJECT_ID"],
    location=os.environ.get("VERTEX_LOCATION", "us-central1"),
    temperature=0,  # deterministic grading, so config diffs aren't judge noise
)

# No embeddings: context_recall, context_precision and faithfulness are all
# LLM-only in ragas 0.2. Only needed if you add answer_relevancy,
# semantic_similarity or answer_correctness.

CONFIGS = [
    {"use_semantic_chunking": False, "use_reranking": False},
    {"use_semantic_chunking": False, "use_reranking": True},
    {"use_semantic_chunking": True, "use_reranking": False},
    {"use_semantic_chunking": True, "use_reranking": True},
]


def run_config(use_semantic_chunking: bool, use_reranking: bool):
    label = (
        f"chunk-{'semantic' if use_semantic_chunking else 'fixed'}"
        f"_rerank-{'on' if use_reranking else 'off'}"
    )
    print(f"\n=== {label} ===")

    store, hybrid = build_pipeline(use_semantic_chunking)

    questions, contexts_list, answers, references = [], [], [], []

    for case in TEST_CASES:
        # No sleep: the old 2s pause was for AI Studio free tier limits.
        retrieved_contexts, generated_answer = run_query(
            case["question"], store, hybrid, use_reranking
        )
        questions.append(case["question"])
        contexts_list.append(retrieved_contexts)
        answers.append(generated_answer)
        references.append(case["reference"])

    # ragas 0.2 column names. The 0.1 names silently produce NaN.
    dataset = Dataset.from_dict({
        "user_input": questions,
        "retrieved_contexts": contexts_list,
        "response": answers,
        "reference": references,
    })

    result = evaluate(
        dataset,
        metrics=[context_recall, context_precision, faithfulness],
        llm=LangchainLLMWrapper(judge_llm),
        run_config=RunConfig(max_workers=MAX_WORKERS, timeout=180),
    )

    df = result.to_pandas()
    print(df)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join(RESULTS_DIR, f"{label}_{stamp}.csv")
    df.to_csv(path, index=False)
    print(f"saved -> {path}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", type=int, choices=range(len(CONFIGS)),
        help="run a single config by index (0-3)",
    )
    args = parser.parse_args()

    selected = [CONFIGS[args.only]] if args.only is not None else CONFIGS

    # No try/except: a traceback names the problem, a caught exception hides it.
    for config in selected:
        run_config(**config)
import argparse
import os
import re
import warnings
from datetime import datetime

import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextPrecision,
    answer_relevancy,
    context_recall,
    faithfulness,
)
from ragas.metrics._context_precision import ContextPrecisionPrompt
from ragas.run_config import RunConfig

from src.eval_data import TEST_CASES
from src.generator import NO_ANSWER
from src.pipeline_run import CANDIDATE_K, TOP_K, build_pipeline, run_query

warnings.filterwarnings("ignore", category=DeprecationWarning)
load_dotenv()

MAX_WORKERS = 4
JUDGE_TIMEOUT = 600
RESULTS_DIR = "results"


# The stock prompt marks a chunk useful only if it supports the whole answer,
# which a ~500-char excerpt of a novel almost never does, so precision collapses to ~0.
class PartialContextPrecisionPrompt(ContextPrecisionPrompt):
    instruction = (
        "Given question, answer and context verify if the context was useful in arriving "
        "at the given answer. The context is a short excerpt from a longer source, so mark "
        'it useful ("1") if it supports at least one part of the answer, even if it does '
        'not cover the whole answer. Give "0" only if it supports no part of the answer. '
        "Give verdict with json output."
    )


context_precision = ContextPrecision(
    name="context_precision_partial",
    context_precision_prompt=PartialContextPrecisionPrompt(),
)

METRICS = [context_recall, context_precision, faithfulness, answer_relevancy]

CONFIGS = [
    {"use_semantic_chunking": False, "use_reranking": False},
    {"use_semantic_chunking": False, "use_reranking": True},
    {"use_semantic_chunking": True, "use_reranking": False},
    {"use_semantic_chunking": True, "use_reranking": True},
]


def build_judge():
    return LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            vertexai=True,
            project=os.environ["VERTEX_PROJECT_ID"],
            location=os.environ.get("VERTEX_LOCATION", "us-central1"),
            temperature=0,
        )
    )


def build_judge_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))


# The generator is told to answer partially when it can, so it often declines in its own
# words ("The passages do not state...") instead of the exact NO_ANSWER sentence.
_DECLINE = re.compile(
    r"\b(?:do|does|did)\s+not\s+(?:state|mention|say|specify|contain|indicate|"
    r"provide|include|discuss|reveal)\b|\bno\s+(?:information|mention)\b",
    re.IGNORECASE,
)


def is_abstention(answer: str) -> bool:
    return answer.strip().lower().startswith(NO_ANSWER.lower().rstrip("."))


def declined_unanswerable(answer: str) -> bool:
    # Only for questions the book cannot answer: on answerable questions the same
    # phrasing usually marks a partial answer, not a refusal.
    return is_abstention(answer) or bool(_DECLINE.search(answer))


def config_label(use_semantic_chunking: bool, use_reranking: bool, read_images: bool = False) -> str:
    return (
        f"chunk-{'semantic' if use_semantic_chunking else 'fixed'}"
        f"_rerank-{'on' if use_reranking else 'off'}"
        f"{'_img' if read_images else ''}"
    )


def run_config(
    use_semantic_chunking, use_reranking, judge, judge_embeddings, top_k, candidate_k,
    read_images=False,
):
    label = config_label(use_semantic_chunking, use_reranking, read_images)
    print(f"\n=== {label} (top_k={top_k}, candidate_k={candidate_k}) ===")

    store, hybrid = build_pipeline(use_semantic_chunking, read_images=read_images)

    records = []
    for case in TEST_CASES:
        contexts, answer = run_query(
            case["question"], store, hybrid, use_reranking,
            top_k=top_k, candidate_k=candidate_k, section_headings=read_images,
        )
        records.append({
            "question": case["question"],
            "type": case["type"],
            "contexts": contexts,
            "answer": answer,
            "reference": case["reference"],
            "expect_abstention": case.get("expect_abstention", False),
            "abstained": is_abstention(answer),
            "declined": declined_unanswerable(answer),
        })

    scorable = [r for r in records if not r["expect_abstention"]]
    dataset = Dataset.from_dict({
        "user_input": [r["question"] for r in scorable],
        "retrieved_contexts": [r["contexts"] for r in scorable],
        "response": [r["answer"] for r in scorable],
        "reference": [r["reference"] for r in scorable],
    })

    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=judge,
        embeddings=judge_embeddings,
        run_config=RunConfig(max_workers=MAX_WORKERS, timeout=JUDGE_TIMEOUT),
    )

    df = result.to_pandas()
    df.insert(0, "type", [r["type"] for r in scorable])
    df["abstained"] = [r["abstained"] for r in scorable]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = os.path.join(RESULTS_DIR, f"{label}_{stamp}.csv")
    df.to_csv(path, index=False)

    metric_names = [m.name for m in METRICS]
    summary = {"config": label}
    for name in metric_names:
        summary[name] = float(df[name].mean()) if name in df else float("nan")
    summary["abstention_rate"] = sum(r["abstained"] for r in scorable) / len(scorable)
    summary["control_abstained"] = all(
        r["declined"] for r in records if r["expect_abstention"]
    )

    print(df[["type"] + metric_names + ["abstained"]].to_string(index=False))
    print(f"saved -> {path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", type=int, choices=range(len(CONFIGS)))
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidate-k", type=int, default=CANDIDATE_K)
    parser.add_argument(
        "--read-images", action="store_true",
        help="read PDF images with Gemini and tag chunks with their section heading",
    )
    args = parser.parse_args()

    selected = [CONFIGS[args.only]] if args.only is not None else CONFIGS

    judge = build_judge()
    judge_embeddings = build_judge_embeddings()

    summaries = []
    for config in selected:
        summaries.append(
            run_config(
                **config,
                judge=judge,
                judge_embeddings=judge_embeddings,
                top_k=args.top_k,
                candidate_k=args.candidate_k,
                read_images=args.read_images,
            )
        )

    summary_df = pd.DataFrame(summaries)
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    summary_path = os.path.join(RESULTS_DIR, f"summary_{stamp}.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n\n================ SUMMARY ================")
    print(summary_df.to_string(index=False))
    print(f"\nsaved -> {summary_path}")

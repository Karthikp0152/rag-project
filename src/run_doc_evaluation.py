"""Evaluate the multi-format pipeline on eval_docs/ under three setups.

  A_text_only          images are not read at all (the pipeline before steps 1-3)
  B_image_reading      images are read into text descriptions (steps 1-2, the default)
  C_images_in_answers  B, plus matching images are shown to the answer model (step 3)

Each answer is scored for correctness against its reference by an LLM judge: 1 correct,
0.5 partial, 0 wrong. RAGAS is not used here because its judge reads only text, so it
cannot check answers that came from a picture. Unanswerable questions score 1 if the
system declines.
"""
import argparse
import os
import random
import time
import warnings
from datetime import datetime

import pandas as pd
from google.genai import errors, types
from pydantic import BaseModel

from src.eval_docs_data import DOC_FILES, DOC_TEST_CASES
from src.generator import GENERATOR_MODEL, RETRYABLE_STATUS, generate_answer, get_client
from src.pipeline_run import build_library, retrieve, select_images
from src.run_evaluation import declined_unanswerable

warnings.filterwarnings("ignore", category=DeprecationWarning)

RESULTS_DIR = "results"
SETUPS = {
    "A_text_only": {"read_images": False, "images_in_answers": False},
    "B_image_reading": {"read_images": True, "images_in_answers": False},
    "C_images_in_answers": {"read_images": True, "images_in_answers": True},
}
SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_PROMPT = (
    "You grade answers from a document question-answering system.\n"
    "Compare the ANSWER with the REFERENCE answer to the QUESTION.\n"
    '- "correct": the answer states all key facts of the reference and contradicts none of '
    "them. Extra correct detail, citations, and different wording are fine.\n"
    '- "partial": the answer states some key facts but misses others, or is right with a '
    "minor error.\n"
    '- "wrong": the answer misses the key facts, contradicts the reference, or says it does '
    "not have the information.\n\n"
    "QUESTION: {question}\nREFERENCE: {reference}\nANSWER: {answer}"
)


class Verdict(BaseModel):
    verdict: str
    reason: str


def judge(question: str, reference: str, answer: str) -> tuple[str, str]:
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    config = types.GenerateContentConfig(
        temperature=0.0, response_mime_type="application/json", response_schema=Verdict
    )
    for attempt in range(5):
        try:
            response = get_client().models.generate_content(
                model=GENERATOR_MODEL, contents=prompt, config=config
            )
            break
        except errors.APIError as exc:
            if getattr(exc, "code", None) not in RETRYABLE_STATUS or attempt == 4:
                raise
            time.sleep(2**attempt + random.random())
    parsed = response.parsed
    if parsed is None or parsed.verdict not in SCORES:
        return "wrong", f"unparseable judge output: {response.text!r}"
    return parsed.verdict, parsed.reason


def run_setup(name: str, read_images: bool, images_in_answers: bool) -> pd.DataFrame:
    print(f"\n=== {name} ===")
    store, hybrid, images = build_library(DOC_FILES, use_semantic_chunking=True, read_images=read_images)
    image_index = None
    if images_in_answers and images:
        from src.image_index import ImageIndex

        image_index = ImageIndex(images)
    print(f"{len(store.chunks)} passages, {len(images) if read_images else 0} images")

    rows = []
    for case in DOC_TEST_CASES:
        question = case["question"]
        contexts = [r["chunk"] for r in retrieve(question, hybrid, use_reranking=True)]
        shown = []
        if image_index is not None:
            from src.ask import image_label
            from src.image_index import answer_png

            chosen = select_images(question, contexts, images, image_index)
            shown = [
                (image_label(i, image), answer_png(image["id"]))
                for i, image in enumerate(chosen, start=1)
            ]
        answer = generate_answer(question, contexts, source_tags=True, images=shown or None)

        if case.get("expect_abstention"):
            declined = declined_unanswerable(answer)
            verdict, reason = ("correct", "declined") if declined else ("wrong", "answered instead of declining")
        else:
            verdict, reason = judge(question, case["reference"], answer)
        rows.append({
            "setup": name, "category": case["category"], "question": question,
            "reference": case["reference"], "answer": answer, "images_sent": len(shown),
            "verdict": verdict, "score": SCORES[verdict], "judge_reason": reason,
        })
        print(f"  [{verdict:7}] {case['category']:12} {question[:70]}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setups", nargs="+", choices=list(SETUPS), default=list(SETUPS))
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    frames = []
    for name in args.setups:
        df = run_setup(name, **SETUPS[name])
        df.to_csv(os.path.join(RESULTS_DIR, f"docs_{name}_{stamp}.csv"), index=False)
        frames.append(df)

    results = pd.concat(frames)
    by_category = results.pivot_table(index="category", columns="setup", values="score", aggfunc="mean")
    answerable = results[results["category"] != "unanswerable"]
    by_category.loc["OVERALL (answerable)"] = answerable.groupby("setup")["score"].mean()
    by_category.loc["images sent (total)"] = results.groupby("setup")["images_sent"].sum()
    by_category = by_category[[s for s in args.setups]]
    summary_path = os.path.join(RESULTS_DIR, f"docs_summary_{stamp}.csv")
    by_category.round(3).to_csv(summary_path)

    print("\n\n================ SUMMARY (mean correctness; 1 = correct) ================")
    print(by_category.round(2).to_string())
    print(f"\nsaved -> {summary_path}")

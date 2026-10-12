import os
import random
import time

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types

load_dotenv()

GENERATOR_MODEL = "gemini-2.5-flash"
NO_ANSWER = "You do not have enough information."
MAX_RETRIES = 5
RETRYABLE_STATUS = (429, 500, 502, 503, 504)

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=os.environ["VERTEX_PROJECT_ID"],
            location=os.environ.get("VERTEX_LOCATION", "us-central1"),
        )
    return _client


HEADING_NOTE = (
    "After its number, a passage may start with its section heading in brackets, like "
    "[Chapter ONE · Daria]. Use the heading as context. In first-person text, a person's "
    "name in the heading usually identifies who \"I\" is in that passage; refer to that "
    "person by name rather than as \"the narrator\".\n"
)


SOURCE_NOTE = (
    "After its number, each passage starts with its source in brackets: the file name, "
    "then its section heading if it has one, like [report.pdf · Refunds]. Use the heading "
    "as context; in first-person text, a person's name in the heading usually identifies "
    "who \"I\" is. When you cite a passage, also name its file, like [2] (report.pdf).\n"
)


def build_prompt(
    question: str,
    chunks: list[str],
    section_headings: bool = False,
    subject: str = "the provided documents",
    source_tags: bool = False,
) -> str:
    passages = "\n\n".join(
        f"[{i}] {chunk.strip()}" for i, chunk in enumerate(chunks, start=1)
    )
    note = SOURCE_NOTE if source_tags else HEADING_NOTE if section_headings else ""
    return (
        f"You answer questions about {subject} using retrieved passages.\n"
        "The passages are excerpts, may be out of order, and may not all be relevant.\n"
        + note
        + "Use only what the passages state. Do not use outside knowledge.\n"
        "Partial evidence is still evidence: answer what the passages support and say "
        "what remains unclear.\n"
        f"Only if the passages support no part of an answer, reply exactly: {NO_ANSWER}\n"
        "Cite the passage numbers you relied on, like [2].\n\n"
        f"Passages:\n{passages}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer(
    question: str,
    chunks: list[str],
    temperature: float = 0.0,
    section_headings: bool = False,
    subject: str = "the provided documents",
    source_tags: bool = False,
) -> str:
    if not chunks:
        return NO_ANSWER

    prompt = build_prompt(
        question, chunks, section_headings=section_headings, subject=subject,
        source_tags=source_tags,
    )
    config = types.GenerateContentConfig(temperature=temperature)

    for attempt in range(MAX_RETRIES):
        try:
            response = get_client().models.generate_content(
                model=GENERATOR_MODEL, contents=prompt, config=config
            )
        except errors.APIError as exc:
            if getattr(exc, "code", None) not in RETRYABLE_STATUS or attempt == MAX_RETRIES - 1:
                raise
            time.sleep(2**attempt + random.random())
            continue
        text = getattr(response, "text", None)
        return text.strip() if text else NO_ANSWER

    return NO_ANSWER


if __name__ == "__main__":
    from src.pipeline_run import build_pipeline, run_query

    # Recommended configuration from the evaluation: semantic chunking + reranker,
    # with PDF images read and chunks tagged with their section heading.
    store, hybrid = build_pipeline(use_semantic_chunking=True, read_images=True)
    question = "What happens between the main characters?"
    contexts, answer = run_query(
        question, store, hybrid, use_reranking=True, section_headings=True
    )
    print("Question:", question)
    print(f"\nRetrieved {len(contexts)} passages")
    print("\nAnswer:", answer)

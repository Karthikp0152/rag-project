import argparse
import os
import re
import sys

from src.documents import SUPPORTED_TYPES
from src.generator import generate_answer
from src.pipeline_run import CANDIDATE_K, TOP_K, build_library, retrieve

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SOURCE_TAG = re.compile(r"^\[([^\]]+)\]")


def cited_sources(answer: str, contexts: list[str]) -> list[str]:
    """The [file · heading] tags of the passages the answer cites, in citation order."""
    seen = []
    for match in _CITATION.finditer(answer):
        for number in match.group(1).split(","):
            index = int(number) - 1
            if 0 <= index < len(contexts):
                tag = _SOURCE_TAG.match(contexts[index])
                label = f"[{index + 1}] {tag.group(1) if tag else 'untitled'}"
                if label not in seen:
                    seen.append(label)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask questions about one or more documents "
        f"({', '.join(sorted(SUPPORTED_TYPES))})."
    )
    parser.add_argument("files", nargs="+", help="documents to load")
    parser.add_argument("--fixed", action="store_true", help="fixed-size chunks instead of semantic")
    parser.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder reranker")
    parser.add_argument("--no-images", action="store_true", help="do not read images with Gemini")
    args = parser.parse_args()

    missing = [f for f in args.files if not os.path.isfile(f)]
    if missing:
        sys.exit(f"File not found: {', '.join(missing)}")

    print(f"Loading {len(args.files)} file(s)... (first load of a file can take a few minutes)")
    store, hybrid = build_library(
        args.files, use_semantic_chunking=not args.fixed, read_images=not args.no_images
    )
    print(f"Ready: {len(store.chunks)} passages. Ask a question, or press Enter to quit.\n")

    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question or question.lower() in ("quit", "exit"):
            break

        selected = retrieve(
            question, hybrid, use_reranking=not args.no_rerank, top_k=TOP_K, candidate_k=CANDIDATE_K
        )
        contexts = [r["chunk"] for r in selected]
        answer = generate_answer(question, contexts, source_tags=True)
        print(f"\n{answer}")
        sources = cited_sources(answer, contexts)
        if sources:
            print("\nSources:\n  " + "\n  ".join(sources))
        print()


if __name__ == "__main__":
    main()

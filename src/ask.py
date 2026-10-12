import argparse
import os
import re
import sys

from src.documents import SUPPORTED_TYPES
from src.generator import generate_answer
from src.pipeline_run import CANDIDATE_K, TOP_K, build_library, retrieve, select_images

# Citations like [2], [1, 3], [I1], or mixed [I1, 2].
_CITATION = re.compile(r"\[((?:I?\d+)(?:\s*,\s*I?\d+)*)\]")
_SOURCE_TAG = re.compile(r"^\[([^\]]+)\]")


def image_label(number: int, image: dict) -> str:
    where = " · ".join(part for part in (image["source"], image.get("heading")) if part)
    page = f", page {image['page']}" if image.get("page") else ""
    return f"[I{number}] {where} (image{page})"


def cited_sources(answer: str, contexts: list[str], image_labels: list[str] = ()) -> list[str]:
    """The [file · heading] tags of the passages and images the answer cites, in order."""
    seen = []
    for match in _CITATION.finditer(answer):
        for ref in match.group(1).split(","):
            ref = ref.strip()
            if ref.startswith("I"):
                index = int(ref[1:]) - 1
                label = image_labels[index] if 0 <= index < len(image_labels) else None
            else:
                index = int(ref) - 1
                label = None
                if 0 <= index < len(contexts):
                    tag = _SOURCE_TAG.match(contexts[index])
                    label = f"[{index + 1}] {tag.group(1) if tag else 'untitled'}"
            if label and label not in seen:
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
    parser.add_argument(
        "--images-in-answers", action="store_true",
        help="find images visually (CLIP) and show the matching images to Gemini when answering",
    )
    args = parser.parse_args()

    missing = [f for f in args.files if not os.path.isfile(f)]
    if missing:
        sys.exit(f"File not found: {', '.join(missing)}")

    print(f"Loading {len(args.files)} file(s)... (first load of a file can take a few minutes)")
    store, hybrid, images = build_library(
        args.files, use_semantic_chunking=not args.fixed, read_images=not args.no_images
    )
    use_images = args.images_in_answers and bool(images)
    image_index = None
    if use_images:
        from src.image_index import ImageIndex

        image_index = ImageIndex(images)  # loads CLIP only when there are images to search
    print(
        f"Ready: {len(store.chunks)} passages"
        + (f", {len(images)} images" if use_images else "")
        + ". Ask a question, or press Enter to quit.\n"
    )

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
        shown, labels = [], []
        if use_images:
            from src.image_index import answer_png

            chosen = select_images(question, contexts, images, image_index)
            labels = [image_label(i, image) for i, image in enumerate(chosen, start=1)]
            shown = [(label, answer_png(image["id"])) for label, image in zip(labels, chosen)]
        answer = generate_answer(question, contexts, source_tags=True, images=shown or None)
        print(f"\n{answer}")
        sources = cited_sources(answer, contexts, labels)
        if sources:
            print("\nSources:\n  " + "\n  ".join(sources))
        print()


if __name__ == "__main__":
    main()

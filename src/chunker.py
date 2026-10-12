def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must not be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        addition = len(word) + (1 if current else 0)
        if current and length + addition > chunk_size:
            chunks.append(" ".join(current))
            carry: list[str] = []
            carry_length = 0
            for previous in reversed(current):
                extra = len(previous) + (1 if carry else 0)
                if carry_length + extra > chunk_overlap:
                    break
                carry.insert(0, previous)
                carry_length += extra
            current = carry
            length = carry_length
            addition = len(word) + (1 if current else 0)
        current.append(word)
        length += addition

    if current:
        chunks.append(" ".join(current))
    return chunks


if __name__ == "__main__":
    from src.loader import load_pdf

    text = load_pdf("data/_OceanofPDF.com_Pretty_Reckless_-_LJ_Shen.pdf")
    chunks = chunk_text(text)
    lengths = sorted(len(c) for c in chunks)
    print(f"Total characters : {len(text)}")
    print(f"No. of chunks    : {len(chunks)}")
    print(f"Mean chunk length: {sum(lengths) / len(lengths):.1f}")
    print(f"Min / Max        : {lengths[0]} / {lengths[-1]}")
    print("\n---First Chunk---")
    print(chunks[0])

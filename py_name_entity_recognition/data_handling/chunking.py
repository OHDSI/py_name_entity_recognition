from langchain_text_splitters import RecursiveCharacterTextSplitter

from py_name_entity_recognition.observability.logging import logger


def chunk_text_with_offsets(
    text: str, chunk_size: int = 2000, chunk_overlap: int = 300
) -> list[tuple[str, int]]:
    """
    Splits a long text into smaller chunks and returns them with their start offsets.

    Uses a RecursiveCharacterTextSplitter for NLP-aware chunking, respecting
    paragraph and sentence boundaries where possible.

    Args:
        text: The text to be chunked.
        chunk_size: The maximum size of each chunk (in characters).
        chunk_overlap: The number of characters of overlap between consecutive chunks.

    Returns:
        A list of tuples, where each tuple contains a text chunk and its
        starting character offset in the original text.
    """
    if len(text) <= chunk_size:
        return [(text, 0)]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)

    results: list[tuple[str, int]] = []
    current_pos = 0
    for chunk in chunks:
        # Find the position of the chunk in the text, starting from the last position.
        # This handles overlapping chunks correctly.
        pos = text.find(chunk, current_pos)
        if pos != -1:
            results.append((chunk, pos))
            # To find the next chunk, we should start our search from a point
            # after the beginning of the current chunk to avoid re-finding the same chunk.
            # Advancing by a small amount is safer than jumping to the end of the chunk
            # due to the nature of the recursive splitter.
            current_pos = pos + 1
        else:
            logger.warning(
                "Could not reliably find chunk in the original text. "
                "This might occur with complex texts or normalization by the splitter. "
                "The chunk will be skipped, which may affect the final output."
            )
    return results

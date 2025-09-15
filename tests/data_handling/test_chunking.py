from py_name_entity_recognition.data_handling.chunking import chunk_text_with_offsets
from py_name_entity_recognition.observability.logging import logger


def test_chunk_text_with_offsets():
    text = "This is a long text that needs to be chunked."
    chunks = chunk_text_with_offsets(text, chunk_size=10, chunk_overlap=2)
    assert len(chunks) > 1
    assert chunks[0][0] == "This is a"
    assert chunks[0][1] == 0
    assert chunks[1][0] == "a long"
    assert chunks[1][1] == 8


def test_chunk_text_with_offsets_no_chunking():
    text = "Short text."
    chunks = chunk_text_with_offsets(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) == 1
    assert chunks[0][0] == text
    assert chunks[0][1] == 0


def test_chunk_text_with_offsets_chunk_not_found(monkeypatch):
    text = "This is a test text."
    # Mock the splitter to return a chunk that is not in the original text
    monkeypatch.setattr(
        "py_name_entity_recognition.data_handling.chunking.RecursiveCharacterTextSplitter.split_text",
        lambda self, text: ["This is a test", "a completely different chunk"],
    )

    logged_warnings = []
    monkeypatch.setattr(logger, "warning", lambda msg: logged_warnings.append(msg))

    chunks = chunk_text_with_offsets(text, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 1
    assert len(logged_warnings) == 1
    assert "Could not reliably find chunk" in logged_warnings[0]

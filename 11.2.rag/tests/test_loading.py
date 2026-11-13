from pathlib import Path

import pytest
from langchain_core.documents import Document

from rag.pipeline import load_documents


@pytest.mark.parametrize("as_string", [False, True])
def test_immediate_txt_files_sorted_preserved_and_nonempty(tmp_path, as_string):
    (tmp_path / "z.txt").write_text("  Café 🐈\n\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("First document\n", encoding="utf-8")
    (tmp_path / "empty.txt").write_text(" \t\n", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("Not a text document", encoding="utf-8")
    (tmp_path / "ignore.TXT").write_text("Uppercase extension", encoding="utf-8")
    nested = tmp_path / "nested.txt"
    nested.mkdir()
    (nested / "hidden.txt").write_text("Do not traverse", encoding="utf-8")
    docs = load_documents(str(tmp_path) if as_string else tmp_path)
    assert all(isinstance(doc, Document) for doc in docs)
    assert [doc.metadata["source"] for doc in docs] == ["a.txt", "z.txt"]
    assert [doc.page_content for doc in docs] == ["First document\n", "  Café 🐈\n\n"]
    assert all(Path(doc.metadata["source"]).name == doc.metadata["source"] for doc in docs)


def test_empty_missing_and_non_directory(tmp_path):
    assert load_documents(tmp_path) == []
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "missing")
    file = tmp_path / "file.txt"
    file.write_text("text", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        load_documents(file)

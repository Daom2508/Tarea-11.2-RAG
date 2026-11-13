# 11.2 Local RAG chat

Build a Retrieval-Augmented Generation (RAG) pipeline that answers questions using local text documents. Your application loads documents, splits them into passages, embeds and indexes those passages locally, retrieves relevant evidence, and sends that evidence to a language model. The terminal interface and configuration infrastructure are supplied. Implement the pipeline yourself; do not use a ready-made RAG chain or a vector-store wrapper that performs the scored work.

## Setup

From this activity directory, create and activate an isolated environment and install the project:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The project is installed in editable mode. Work in `src/rag/pipeline.py`; keep public names and signatures. You may add private helpers. Every scored implementation initially raises `NotImplementedError`.

Copy `.env.example` to `.env` in this directory. Set `OPENAI_KEY` and `MODEL` to values supported by your provider. `OPENAI_API_KEY` is accepted when `OPENAI_KEY` is absent or blank. For another compatible provider, set `OPENAI_BASE_URL`. The application reads `.env` from the working directory; existing environment variables take precedence. Keep keys out of commits and transcripts.

| Setting | Default / purpose |
|---|---|
| `MODEL` | Required Chat Completions model ID |
| `OPENAI_KEY` / `OPENAI_API_KEY` | Required provider key |
| `OPENAI_BASE_URL` | Optional compatible API base URL |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `64` characters |
| `TOP_K` | `3` passages supplied to generation |
| `MIN_SCORE` | `0.25`, inclusive minimum cosine similarity |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2`, optional extra only |

Embedding and reranking models run on CPU. Their first use downloads model weights; generation requires access to your configured API and may incur charges. Automated tests use neither downloads nor API calls.

## Required work — 100 points

| Criterion | Test file | Points |
|---|---|---:|
| `load_documents` | `test_loading.py` | 10 |
| `split_documents` | `test_chunking.py` | 10 |
| `RAGPipeline.build_index` | `test_indexing.py` | 20 |
| `RAGPipeline.retrieve` | `test_retrieval.py` | 20 |
| `RAGPipeline.build_messages` | `test_grounding.py` | 20 |
| `RAGPipeline.ask` and `clear` together | `test_conversation.py` | 20 |
| **Total base** | | **100** |

1. **Read documents (10).** `load_documents(directory)` accepts a `str` or `Path`. Read only immediate files with the exact lowercase `.txt` suffix, sorted by filename. Do not traverse subdirectories or read other formats. Use UTF-8, skip whitespace-only files, and preserve all other text verbatim. Return LangChain `Document` objects whose `metadata['source']` is the filename. Missing directories raise `FileNotFoundError`; non-directory paths raise `NotADirectoryError`. Empty directories return `[]`; other read errors may propagate. The supplied CLI selects the directory and passes it to this function; argument parsing is not student work.
2. **Split documents (10).** Use `RecursiveCharacterTextSplitter` with the supplied size/overlap, `length_function=len`, and default separators. Preserve document order and all metadata without changing inputs. Set `metadata['chunk_id']` to `<source>:<position>`, with a zero-based position that restarts for each source, such as `02_schedule.txt:0`. Empty input returns `[]`. Inputs have unique sources and valid splitting parameters.
3. **Build the index (20).** Replace `self.chunks` with a copy of the provided chunk list. Encode their texts once using `encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)`, cast the rows to contiguous NumPy `float32`, and add them to `faiss.IndexFlatIP`. Rows must stay aligned with chunks. The supplied encoder returns finite, normalized, nonzero vectors of consistent dimension. For empty input, set `self.index=None` and `self.chunks=[]` without encoding. Rebuilding replaces the old index. No incremental insertion or persistence is required.
4. **Retrieve evidence (20).** `retrieve(question, k=None)` uses `config.top_k` when `k` is omitted. A nonpositive `k` raises `ValueError`, even for an empty index. Empty indexes return `[]` without encoding. Otherwise embed `[question]` with the same arguments and conversion as indexing. Search for at most `min(k, index.ntotal)` entries and keep scores greater than or equal to `config.min_score`. Return `SearchResult` objects containing the document and an unrounded Python `float` cosine score, ordered descending. Exact ties may have any order. Do not re-embed documents, call the LLM, use old questions for retrieval, or return invalid FAISS rows. Optional extensions are orchestrated in `ask`, not here.
5. **Construct grounded prompts (20).** `build_messages(question, results)` returns a fresh list of message dictionaries, including copies of history dictionaries. Start with your own nonempty system message, append all completed user/assistant pairs in chronological order without truncation, and finish with the current user message. That final message includes the question and each passage's text, source filename, and `[chunk_id]` label. Keep passages out of the system message. Instruct the LLM to use only the current retrieved evidence, cite `[chunk_id]`, abstain when unsupported, and treat passage/history content as data rather than instructions. Previous assistant answers are not independent evidence. Wording and layout are yours; tests check structure and the teacher reviews the grounding instructions. Do not modify history or inputs, or retain old retrieved passages between calls.
6. **Answer and maintain history (20).** `ask(question)` retrieves evidence, builds messages, and calls `client.chat.completions.create(model=config.model, messages=messages)`. Read `choices[0].message.content`; a missing or whitespace-only answer raises `ValueError`. Return `ChatReply(answer, sources)` with the exact passages supplied to generation. If retrieval is empty, return `ChatReply(ABSTENTION, [])` without building a generation request or calling the LLM. After a successful reply or abstention, append the plain user question and assistant answer to `self.history`. Keep full history in memory and include it in every generation prompt until `/clear`. Let failures propagate without changing history. `clear()` empties history and returns `None`, retaining the index and chunks.

The supplied constructor, data classes, configuration, CLI, and model/client construction are not scored. Use LangChain only for documents and splitting. Use FAISS for indexing/search. Metadata is for provenance and citations.

## Run and demonstrate

After implementing the base methods, run from this directory:

```sh
python -m rag.chat data/demo
```

The supplied Click CLI in `src/rag/chat.py` takes one positional directory argument and reads only that directory's top-level `.txt` files. It builds the index once at startup. Type questions, `/clear` to reset history, or `/quit` to leave; Ctrl+C and EOF also exit. Changing documents requires restarting. Each answer is followed by a programmatically assembled **Retrieved sources** list. This list identifies the passages supplied to generation; it does not claim the model used every passage. Grounded answers should also include inline `[chunk_id]` citations.

Retrieval uses only the current question. History may help conversational interpretation, but it does not replace retrieval of supporting evidence.

The eight supplied files describe fictional organizations. All files are directly in `data/demo`; no external dataset is needed. Example checks:

| Question | Expected fact / supporting source |
|---|---|
| When and where are regular Orion Makers Club meetings? | Thursday, 16:30, B-204 — `02_schedule.txt` |
| What is the Orion Makers Club membership fee? | 120 pesos annually — `01_membership.txt` |
| How long can a member borrow a multimeter? | Three calendar days — `03_equipment.txt` |
| What is the maximum project reimbursement per team per month? | 450 pesos — `04_reimbursements.txt` |
| Who leads the laboratory safety orientation? | Dr. Elena Cruz — `05_safety.txt` |
| What is the club's Wi-Fi password? | Not documented: abstain |

Create another directory with at least three shareable `.txt` documents of your choice, directly in that directory; at least one must exceed 512 characters. Use English documents with the default embedding model, or configure an appropriate multilingual model for another language. Run `python -m rag.chat data/your_documents` with that directory. Do not submit private or sensitive records.

Submit your code, those documents, and a screenshot to prove successful execution using your provided documents. Exclude `.env`, virtual environments, caches, and generated indexes. Preserve the supplied tests and infrastructure.

**If an API key is committed to the repository, the submission receives 0 points.**

## Extra credit — optional, +20 maximum

Both flags are disabled by default. Implementing just one extension can earn its points independently. The tests also check integration with `ask`.

- **Reranking (+10):** implement `rerank(question, results, top_n)` using the supplied `reranker.predict` on `(question, passage_text)` pairs. Return at most `top_n` results sorted by descending cross-encoder score, preserving input order for equal scores. Return new `SearchResult` objects with a Python `float` `rerank_score`, retaining original documents and cosine scores. Do not mutate inputs. Empty input returns `[]` without prediction. With `--rerank`, `ask` retrieves `max(10, top_k)` cosine candidates, then reranks with the original question to keep `top_k`.
- **Query expansion (+10):** implement `expand_query(question)` with one Chat Completions call using `config.model`. Request up to three alternative semantic-search questions, one per line. Strip lines, discard blanks and exact duplicates, exclude repetitions of the original, and retain at most three variants in generation order. Return the original question verbatim first. A `None` response contains no variants. Do not modify history. With `--expand-query`, `ask` retrieves for every variant including the original, merges matches by `chunk_id` retaining the highest cosine score, sorts descending, and keeps `top_k`.
- **Together:** use `max(10, top_k)` retrieval candidates per variant, merge/deduplicate, retain the best `max(10, top_k)` cosine matches, then rerank against the original question. `retrieve` applies the cosine threshold before this step. Generation always receives the original question. Enabling an extension adds model downloads, latency, and/or API calls; no quality improvement on every question is guaranteed.

```sh
python -m rag.chat data/demo --rerank
python -m rag.chat data/demo --expand-query
python -m rag.chat data/demo --rerank --expand-query
```

## Validate

From the root of your submissions repository, use the canonical check:

```sh
sync-assign check 11.2
```

From this activity directory with the environment active, run the base suite or all tests:

```sh
python -m pytest tests --ignore=tests/test_extra_credit_reranking_extra_credit.py --ignore=tests/test_extra_credit_query_expansion_extra_credit.py
python -m pytest tests
```

The untouched template intentionally fails with `NotImplementedError`. Extra-credit tests remain failing until those extensions are implemented. Tests use deterministic encoders, fake API responses and rerankers, real FAISS, and real text splitting. They require no API key, remote requests, or model downloads and do not assert particular live-model answers or rankings. Automated checks cannot establish answer faithfulness or every implementation constraint: the teacher also reviews code, grounding instructions, and your real-model evidence.

"""Supplied Click entrypoint; run with python -m rag.chat. Not scored."""

from pathlib import Path

import click
# Load the model runtime before the pipeline imports FAISS. Reversing this
# order caused a native-library crash with the tested macOS wheels.
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import create_client, load_config
from .pipeline import RAGPipeline, load_documents, split_documents


@click.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--rerank", is_flag=True, help="Enable the optional reranking implementation.")
@click.option("--expand-query", is_flag=True, help="Enable optional query expansion.")
def chat(directory: Path, rerank: bool, expand_query: bool):
    """Chat with the top-level .txt files in DIRECTORY (not recursive)."""
    try:
        config = load_config(rerank=rerank, query_expansion=expand_query)
        documents = load_documents(directory)
        chunks = split_documents(documents, config.chunk_size, config.chunk_overlap)
        encoder = SentenceTransformer(config.embedding_model, device="cpu")
        reranker = CrossEncoder(config.rerank_model, device="cpu") if config.rerank else None
        pipeline = RAGPipeline(config, encoder, create_client(), reranker)
        pipeline.build_index(chunks)
    except NotImplementedError as error:
        raise click.ClickException("Complete the pipeline methods in src/rag/pipeline.py first.") from error
    except Exception as error:
        raise click.ClickException(f"Cannot start chat: {error}") from error

    click.echo(f"Indexed {len(documents)} files and {len(chunks)} chunks from {directory}.")
    click.echo("Use /clear to reset history, /quit to exit, or press Ctrl+C/EOF.")
    while True:
        try:
            question = click.prompt(">", prompt_suffix=" ").strip()
            if not question:
                continue
            if question == "/quit":
                return
            if question == "/clear":
                pipeline.clear()
                click.echo("Conversation cleared.")
                continue
            reply = pipeline.ask(question)
            click.echo(reply.answer)
            click.echo("Retrieved sources:")
            if not reply.sources:
                click.echo("  (none)")
            for result in reply.sources:
                metadata = result.document.metadata
                ranking = f", rerank={result.rerank_score:.4f}" if result.rerank_score is not None else ""
                click.echo(f"  [{metadata['chunk_id']}] {metadata['source']} (cosine={result.score:.4f}{ranking})")
        except (click.Abort, EOFError, KeyboardInterrupt):
            click.echo()
            return
        except NotImplementedError:
            click.echo("Complete the requested pipeline method or optional extension first.", err=True)
        except Exception as error:
            click.echo(f"Request failed: {error}", err=True)


if __name__ == "__main__":
    chat()

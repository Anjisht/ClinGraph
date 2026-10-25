"""Script to ingest PubMed articles into ClinGraph."""

import argparse
import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.core.ids import generate_document_id
from app.core.logging import logger
from app.db.session import SessionFactory, init_db
from app.db.store import DocumentStore
from app.ingestion.chunker import chunk_document
from app.ingestion.ncbi_client import NCBIClient
from app.ingestion.parser import parse_pubmed_xml
from app.retrieval.embedder import get_embedder


async def main():
    parser = argparse.ArgumentParser(description="Ingest PubMed literature into ClinGraph database")
    parser.add_argument(
        "--query", type=str, help="Search query for PubMed (e.g. 'type 2 diabetes')"
    )
    parser.add_argument("--pmids", type=str, nargs="+", help="Explicit list of PMIDs to fetch")
    parser.add_argument(
        "--sample-file", type=str, help="Path to local XML file for offline ingestion"
    )
    parser.add_argument("--max-results", type=int, default=5, help="Maximum articles to fetch")
    args = parser.parse_args()

    settings = get_settings()
    init_db()

    xml_content = ""
    if args.sample_file:
        file_path = Path(args.sample_file)
        if not file_path.exists():
            logger.error(f"Sample file not found: {args.sample_file}")
            return
        xml_content = file_path.read_text(encoding="utf-8")
        logger.info(f"Loaded XML from sample file: {args.sample_file}")
    elif args.query or args.pmids:
        client = NCBIClient(email=settings.NCBI_EMAIL, api_key=settings.NCBI_API_KEY)
        pmids = args.pmids or []
        if args.query and not pmids:
            logger.info(f"Searching PubMed for '{args.query}'...")
            pmids = await client.esearch(args.query, retmax=args.max_results)

        if not pmids:
            logger.warning("No PMIDs found.")
            return

        logger.info(f"Fetching XML for {len(pmids)} article(s): {pmids}")
        xml_content = await client.efetch_xml(pmids)
    else:
        logger.error("Must provide either --query, --pmids, or --sample-file")
        return

    docs = parse_pubmed_xml(xml_content)
    logger.info(f"Parsed {len(docs)} document(s)")

    embedder = get_embedder(model_name=settings.EMBEDDING_MODEL)

    with SessionFactory() as session:
        store = DocumentStore(session)
        total_chunks = 0
        for doc_base in docs:
            clean_pmid = doc_base.source_id.replace("PMID:", "")
            doc_id = generate_document_id(prefix=f"pmid_{clean_pmid}")

            full_doc, chunks = chunk_document(doc_base, document_id=doc_id)
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_texts(texts)
            for ch, emb in zip(chunks, embeddings, strict=False):
                ch.embedding = emb

            store.save_document(full_doc, chunks)
            total_chunks += len(chunks)
            logger.info(
                f"Ingested {doc_base.source_id}: {doc_base.title[:60]}... ({len(chunks)} chunks)"
            )

        logger.info(f"Successfully ingested {len(docs)} document(s) and {total_chunks} chunk(s).")


if __name__ == "__main__":
    asyncio.run(main())

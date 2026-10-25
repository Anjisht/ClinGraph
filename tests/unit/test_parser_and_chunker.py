"""Unit tests for PubMed XML parsing and document chunking."""

from pathlib import Path

from app.ingestion.chunker import chunk_document, compute_content_hash
from app.ingestion.parser import parse_pubmed_xml
from app.schemas.document import DocumentBase

SAMPLE_XML_PATH = Path("data/samples/pubmed_sample.xml")


def test_parse_pubmed_sample_xml():
    xml_text = SAMPLE_XML_PATH.read_text(encoding="utf-8")
    docs = parse_pubmed_xml(xml_text)
    assert len(docs) == 2

    doc1 = docs[0]
    assert doc1.source_id == "PMID:31234567"
    assert "SGLT2 Inhibitors" in doc1.title
    assert len(doc1.authors) == 2
    assert "Smith John A" in doc1.authors
    assert doc1.publication_date == "2023-05-12"
    assert "BACKGROUND:" in doc1.abstract
    assert "RESULTS:" in doc1.abstract
    assert doc1.tenant_id == "public"

    doc2 = docs[1]
    assert doc2.source_id == "PMID:32345678"
    assert "GLP-1 Receptor Agonists" in doc2.title


def test_parse_malformed_xml():
    malformed = "<NotPubmed><Broken"
    docs = parse_pubmed_xml(malformed)
    assert docs == []


def test_parse_missing_abstract():
    xml = """
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>99999999</PMID>
          <Article>
            <ArticleTitle>Title Without Abstract</ArticleTitle>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    docs = parse_pubmed_xml(xml)
    assert len(docs) == 1
    assert docs[0].source_id == "PMID:99999999"
    assert docs[0].abstract is None


def test_chunking_with_character_offsets():
    doc_base = DocumentBase(
        source="pubmed",
        source_id="PMID:100",
        title="Test Document",
        abstract=(
            "First sentence here. Second sentence with findings. Third sentence regarding outcomes."
        ),
    )
    doc, chunks = chunk_document(
        doc_base, document_id="doc_test_100", target_chars=50, overlap_chars=10
    )

    assert len(chunks) >= 2
    full_text = doc_base.abstract

    for c in chunks:
        assert c.document_id == "doc_test_100"
        assert c.chunk_id.startswith("doc_test_100_c")
        # Offset integrity verification
        extracted = full_text[c.char_start : c.char_end]
        assert extracted == c.text
        assert c.token_count > 0

    assert doc.content_hash == compute_content_hash(full_text)

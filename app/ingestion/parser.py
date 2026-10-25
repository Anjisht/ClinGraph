"""Robust XML parser for PubMed bibliographic records and abstracts."""

import xml.etree.ElementTree as ET

from app.core.logging import logger
from app.schemas.document import DocumentBase


def _extract_text(element: ET.Element | None) -> str:
    """Extract and join all inner text recursively, stripping formatting tags."""
    if element is None:
        return ""
    return "".join(element.itertext()).strip()


def parse_pubmed_article(article_el: ET.Element) -> DocumentBase | None:
    """Parse a single PubmedArticle XML element into DocumentBase."""
    try:
        medline = article_el.find("MedlineCitation")
        if medline is None:
            return None

        pmid_el = medline.find("PMID")
        if pmid_el is None or not pmid_el.text:
            return None
        pmid = pmid_el.text.strip()

        article = medline.find("Article")
        if article is None:
            return None

        # Title
        title_el = article.find("ArticleTitle")
        title = _extract_text(title_el) or "Untitled Document"

        # Authors
        authors: list[str] = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = _extract_text(author.find("LastName"))
                fore_name = _extract_text(author.find("ForeName")) or _extract_text(
                    author.find("Initials")
                )
                if last_name:
                    authors.append(f"{last_name} {fore_name}".strip())
                elif fore_name:
                    authors.append(fore_name)

        # Publication Date
        pub_date_str: str | None = None
        pub_date_el = article.find(".//JournalIssue/PubDate")
        if pub_date_el is not None:
            year = _extract_text(pub_date_el.find("Year"))
            month = _extract_text(pub_date_el.find("Month"))
            day = _extract_text(pub_date_el.find("Day"))
            if year:
                parts = [year]
                if month:
                    parts.append(month)
                if day:
                    parts.append(day)
                pub_date_str = "-".join(parts)
            else:
                medline_date = _extract_text(pub_date_el.find("MedlineDate"))
                if medline_date:
                    pub_date_str = medline_date

        # Abstract
        abstract_parts: list[str] = []
        abstract_el = article.find("Abstract")
        if abstract_el is not None:
            for text_node in abstract_el.findall("AbstractText"):
                label = text_node.get("Label") or text_node.get("NlmCategory")
                text = _extract_text(text_node)
                if text:
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)

        abstract = "\n\n".join(abstract_parts) if abstract_parts else None

        source_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        return DocumentBase(
            source="pubmed",
            source_id=f"PMID:{pmid}",
            title=title,
            authors=authors,
            publication_date=pub_date_str,
            abstract=abstract,
            license="public-domain-nlm",
            source_url=source_url,
            tenant_id="public",
            allowed_roles=["*"],
            allowed_groups=[],
            permission_revision=1,
        )
    except Exception as exc:
        logger.warning(f"Error parsing PubMed article element: {str(exc)}")
        return None


def parse_pubmed_xml(xml_content: str) -> list[DocumentBase]:
    """Parse a full PubMed XML response containing multiple articles."""
    if not xml_content.strip():
        return []

    documents: list[DocumentBase] = []
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        logger.error(f"Malformed XML payload from PubMed: {str(exc)}")
        return []

    # Handle PubmedArticleSet
    for article in root.findall(".//PubmedArticle"):
        doc = parse_pubmed_article(article)
        if doc is not None:
            documents.append(doc)

    return documents

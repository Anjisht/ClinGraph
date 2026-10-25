# ClinGraph Data Directory

This directory stores local data fixtures and sample evaluation sets.

## Invariants

- **No raw private patient data or unverified full-text corpora** may be committed to this repository.
- Use synthetic clinical vignettes and public PubMed abstracts with license and provenance attribution.
- The `samples/` subfolder contains minimal fixtures for testing parser, retrieval, and verification pipelines without external network calls.

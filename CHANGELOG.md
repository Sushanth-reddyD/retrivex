# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-10-07

### Added
- Initial release of RetriVex
- Core neighbor-expansion retrieval functionality
- Intelligent multi-factor scoring (similarity, adjacency, continuity)
- Edge-balanced ordering to mitigate U-shaped positional bias
- Span composition and de-duplication
- Simple and sentence-based chunking utilities
- Comprehensive test suite (32 unit tests)
- Documentation:
  - Comprehensive README with usage examples
  - Concepts guide explaining positional bias
  - Contributing guidelines
  - Quickstart example
- CI/CD pipeline with GitHub Actions
- Support for Python 3.10+

### Features
- Configurable neighbor window (±L expansion)
- Multiple ordering strategies (edge_balanced, score_desc, front_first, back_first)
- Token budget management
- Heading boundary detection
- Semantic continuity checking
- Provenance tracking for citations

### Package Structure
- `retrivex.core`: Core models and retrieval logic
- `retrivex.utils`: Chunking utilities
- `retrivex.adapters`: Framework adapters (planned)
- `retrivex.eval`: Evaluation utilities (planned)

## [Unreleased]

### Planned for v0.2
- Structure-aware expansion (heading-bounded, page-bounded)
- LangChain BaseRetriever adapter
- LlamaIndex BaseRetriever adapter
- Enhanced parent document support

### Planned for v0.3
- Late-interaction rerank hook (ColBERT/ColBERTv2)
- Advanced evaluation metrics
- Performance benchmarks vs. existing approaches

### Planned for v0.4
- Answer-shape policies (regex-based expansion)
- Haystack adapter
- Interactive tuning guide

### Planned for v0.5
- Go adapter for Qdrant/PGVector microservices
- REST API server
- Performance optimizations

[0.1.0]: https://github.com/Sushanth-reddyD/retrivex/releases/tag/v0.1.0

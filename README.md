# RetriVex 🔍

**Lightweight neighbor-expansion retriever for addressing LLM positional bias in long contexts**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## What is RetriVex?

RetriVex is a drop-in wrapper around your existing vector retriever that intelligently expands neighbors (±L chunks), merges contiguous spans, scores them intelligently, and orders the final context to address the well-documented **U-shaped positional bias** in LLMs.

### The Problem: Lost-in-the-Middle

LLMs suffer from U-shaped attention patterns—they over-attend to the beginning and end of context while missing crucial information in the middle. This is especially problematic in RAG systems where relevant chunks might be scattered throughout retrieved context.

### The Solution

RetriVex addresses this through:

1. **Neighbor Expansion**: Extends retrieved chunks with ±L adjacent chunks for better context continuity
2. **Intelligent Scoring**: Combines similarity, adjacency, semantic continuity, and parent bonuses
3. **Edge-Balanced Ordering**: Places high-value content at edges where LLMs attend better
4. **Span Composition**: Merges contiguous chunks to reduce fragmentation

## Quick Start

### Installation

```bash
pip install retrivexlib
```

### Basic Usage

```python
from retrivex import SpanComposer, RetrievalConfig, SeedHit, Chunk, ChunkMetadata

# Configure retrieval
config = RetrievalConfig(
    window=2,              # Expand ±2 neighbors around each hit
    k=6,                   # Use top-6 seed hits
    token_budget=2000,     # Max 2000 tokens in final context
    ordering_strategy="edge_balanced"  # Counter U-shaped bias
)

# Initialize composer
composer = SpanComposer(config)

# Your seed hits from vector search
seed_hits = [
    SeedHit(chunk=my_chunk, similarity_score=0.95),
    # ... more hits
]

# Retrieve and compose spans
spans = composer.retrieve(
    seed_hits=seed_hits,
    chunk_store=chunk_store,  # Dict[(doc_id, chunk_id)] -> Chunk
)

# Use composed context
context = "\n\n".join(span.text for span in spans)
```

## Key Features

### 🎯 Smart Neighbor Expansion

Automatically expands each retrieved chunk with surrounding neighbors, respecting:
- Document boundaries
- Heading changes (optional)
- Semantic continuity thresholds

### 📊 Multi-Factor Scoring

Each span is scored using:
- **Similarity** (w₀=1.0): Base relevance from vector search
- **Adjacency** (w₁=0.6): Exponential decay with distance from seed hits
- **Continuity** (w₂=0.2): Semantic flow between adjacent chunks
- **Parent Bonus** (w₃=0.1): Bonus for including parent sections

### 🔄 Edge-Balanced Ordering

Places highest-value content at beginning and end of context to exploit LLM attention patterns:

```
[High-value span 1]  ← Front (high attention)
[Medium-value span]  ← Middle (lower attention)
[High-value span 2]  ← Back (high attention)
```

### 🧩 Automatic De-duplication

Intelligently handles overlapping spans, favoring longer and higher-scoring ones.

## Configuration

```python
config = RetrievalConfig(
    # Core parameters
    window=2,                    # ±L neighbors (default: 2)
    k=6,                         # Base kNN hits (default: 6)
    token_budget=2000,           # Max tokens (default: 2000)
    
    # Scoring weights
    similarity_weight=1.0,       # w₀: base similarity
    adjacency_weight=0.6,        # w₁: neighbor proximity
    continuity_weight=0.2,       # w₂: semantic flow
    parent_weight=0.1,           # w₃: parent bonus
    
    # Advanced
    distance_decay_beta=0.7,     # β: adjacency decay rate
    min_continuity=0.5,          # Min similarity to cross headings
    max_spans=None,              # Limit number of spans
    stop_on_heading_change=True, # Respect section boundaries
    
    # Ordering strategy
    ordering_strategy="edge_balanced",  # or "score_desc", "front_first", "back_first"
)
```

## Comparison with Existing Approaches

| Feature | RetriVex | Parent-Document | Sentence-Window | Auto-Merging |
|---------|----------|-----------------|-----------------|--------------|
| **Focus** | Neighbor expansion + position bias | Hierarchical retrieval | Sentence context | Hierarchical merging |
| **Complexity** | Low | Medium | Low | High |
| **Position-aware ordering** | ✅ Yes (edge-balanced) | ❌ No | ❌ No | ❌ No |
| **Multi-factor scoring** | ✅ Yes | ❌ No | ❌ No | ⚠️ Limited |
| **Setup effort** | Minimal | Medium | Low | High |
| **Best for** | Long-context QA | Document-level tasks | Precise answers | Complex hierarchies |

**When to use RetriVex:**
- Answers span multiple adjacent chunks
- Using long-context models (8k+ tokens)
- Need to mitigate lost-in-the-middle
- Want a lightweight, transparent solution

**When to use alternatives:**
- Parent-Document: When answers are always at document level
- Sentence-Window: For very precise, single-sentence answers
- Auto-Merging: Complex hierarchical documents (legal, technical)

## Chunking Best Practices

### Fixed-Size Chunking (Recommended for Most Cases)

```python
from retrivex.utils import SimpleChunker

chunker = SimpleChunker(
    chunk_size=200,    # tokens
    overlap=50,        # token overlap
)

chunks = chunker.chunk_text(
    text=document_text,
    doc_id="doc_123",
    heading_path=["Chapter 1", "Section A"],
)
```

### Sentence-Based Chunking

```python
from retrivex.utils import SentenceChunker

chunker = SentenceChunker(
    sentences_per_chunk=3,
    overlap_sentences=1,
)

chunks = chunker.chunk_text(text, doc_id="doc_123")
```

### Metadata Requirements

Each chunk needs:
- `doc_id`: Document identifier
- `chunk_id`: Monotonic within document (0, 1, 2, ...)
- `char_start`, `char_end`: Character positions
- `heading_path` (optional): Section hierarchy
- `parent_id` (optional): Parent section reference

## Performance Characteristics

### Time Complexity
- Neighbor expansion: O(k × L) where k = seed hits, L = window size
- Span composition: O(k × L × log(k × L))
- Scoring: O(n × k) where n = spans
- **Total**: O(k × L × log(k × L)) - very efficient!

### Space Complexity
- O(k × L) for expanded chunks
- Minimal memory footprint compared to alternatives

### Typical Performance
- **Retrieval latency**: +5-10ms over base kNN (negligible)
- **Improvement**: +5-15% absolute recall on adjacent-answer cases
- **Token efficiency**: Same or better F1 at fixed token budget

## Benchmark Results

RetriVex has been evaluated against **LongBench**, **RULER**, and **NIAH** benchmarks with 25 samples each:

### Performance Summary

| Method | Avg F1 Score | Avg Recall@k | Speed vs Vanilla k-NN |
|--------|--------------|--------------|----------------------|
| **Vanilla k-NN** | 0.067 | 0.667 | Baseline |
| **Simple Expansion** | 0.066 | 0.613 | 6% slower |
| **RetriVex (Default)** | **0.073** | **0.713** | **16% faster** |
| **RetriVex (Optimized)** | **0.074** | **0.667** | **17% faster** |

### Key Findings

- **🚀 Speed Improvement**: 16-50% faster than vanilla k-NN across benchmarks
- **📈 Quality Maintenance**: Equivalent or better F1/recall scores
- **🎯 Position Robustness**: Reduced position bias through edge-balanced ordering
- **⚡ Token Efficiency**: Better context utilization with moderate token increase

### Benchmark-Specific Results

#### LongBench (Multi-task QA)
- **Quality**: Perfect recall (1.0) and MRR (1.0) across all methods
- **Speed**: RetriVex achieves **50% speed improvement** (0.035ms vs 0.070ms)
- **Tokens**: 69% more tokens but significantly faster processing

#### RULER (Position Sensitivity)  
- **Challenge**: All methods struggle with F1 scoring (0.0 across board)
- **Exact Match**: Variable performance (12-64% depending on configuration)
- **Insight**: Synthetic benchmarks may not reflect real-world scenarios

#### NIAH (Needle in Haystack)
- **Quality**: RetriVex shows better recall (0.92 vs 0.84) and MRR (0.92 vs 0.58)
- **Position Bias**: Significantly reduced (-0.02 vs -0.52)
- **Consistency**: More balanced performance across document positions

### Configuration Recommendations

| Use Case | Configuration | Expected Benefit |
|----------|---------------|------------------|
| **Speed-Critical** | `window=1, budget=2000` | 50% speed improvement |
| **Quality-Focused** | `window=3, budget=4000` | Best position handling |
| **Balanced** | `window=2, budget=2000` | Optimal speed-quality trade-off |

📊 **Detailed Results**: See [EVALUATION_REPORT.md](EVALUATION_REPORT.md) for comprehensive analysis.

## Evaluation Metrics

RetriVex is designed to be evaluated on:

1. **Recall@k_span**: % queries where merged span contains answer
2. **Rescued-by-neighbor %**: How often neighbors (not seeds) had the answer
3. **EM/F1**: End-to-end QA accuracy at fixed token budget
4. **Position sensitivity**: Performance vs. answer position in context

## Roadmap

- **v0.2**: Structure-aware expansion (heading-bounded, page-bounded)
- **v0.3**: Late-interaction rerank hook (ColBERT/ColBERT-v2 support)
- **v0.4**: Answer-shape policies (regex-based expansion)
- **v0.5**: Go adapter for Qdrant/PGVector microservices

## Framework Integration

### LangChain (Coming Soon)

```python
from retrivex.adapters.langchain import RetriVexRetriever

retriever = RetriVexRetriever(
    base_retriever=your_vector_store.as_retriever(),
    config=config,
)
```

### LlamaIndex (Coming Soon)

```python
from retrivex.adapters.llamaindex import RetriVexRetriever

retriever = RetriVexRetriever(
    base_retriever=your_index.as_retriever(),
    config=config,
)
```

## Research Foundation

RetriVex is built on established research:

- [Lost in the Middle](https://arxiv.org/abs/2307.03172): Documents U-shaped attention bias
- [Found in the Middle](https://arxiv.org/abs/2311.09198): Positional calibration methods
- [LongBench](https://arxiv.org/abs/2308.14508): Long-context evaluation benchmark
- [RULER](https://arxiv.org/abs/2404.06654): Synthetic length extrapolation benchmark

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

If you use RetriVex in your research, please cite:

```bibtex
@software{retrivexlib2024,
  title = {RetriVexLib: Neighbor-Expansion Retriever for Long-Context LLMs},
  author = {Sushanth Reddy},
  year = {2024},
  url = {https://github.com/Sushanth-reddyD/retrivex}
}
```

## Acknowledgments

Built with inspiration from:
- LangChain's Parent Document Retriever
- LlamaIndex's Sentence Window and Auto-Merging retrievers
- Research on positional bias in LLMs

---

**Note**: The PyPI package name is `retrivexlib` to avoid conflicts. The project is still called RetriVex.
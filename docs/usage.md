# Usage Guide

This guide walks you through using RetriVex in your RAG application.

## Table of Contents

1. [Installation](#installation)
2. [Basic Workflow](#basic-workflow)
3. [Chunking Your Documents](#chunking-your-documents)
4. [Setting Up Vector Search](#setting-up-vector-search)
5. [Configuring RetriVex](#configuring-retrivex)
6. [Using Retrieved Spans](#using-retrieved-spans)
7. [Integration Examples](#integration-examples)
8. [Performance Tuning](#performance-tuning)
9. [Common Patterns](#common-patterns)

## Installation

```bash
pip install retrivex
```

Optional dependencies:

```bash
# For LangChain integration (coming in v0.2)
pip install retrivex[langchain]

# For LlamaIndex integration (coming in v0.2)
pip install retrivex[llamaindex]

# For development
pip install retrivex[dev]
```

## Basic Workflow

### 1. Import RetriVex

```python
from retrivex import (
    SpanComposer,
    RetrievalConfig,
    SeedHit,
    Chunk,
    ChunkMetadata,
)
from retrivex.utils import SimpleChunker
```

### 2. Chunk Your Documents

```python
# Initialize chunker
chunker = SimpleChunker(
    chunk_size=200,    # ~200 tokens per chunk
    overlap=50,        # 50 token overlap
)

# Chunk a document
chunks = chunker.chunk_text(
    text=your_document_text,
    doc_id="doc_123",
    heading_path=["Chapter 1", "Introduction"],  # Optional
)
```

### 3. Create Embeddings and Store Chunks

```python
import openai  # or your preferred embedding provider

chunk_store = {}
for chunk in chunks:
    # Generate embedding (use your actual embedding model)
    embedding = openai.Embedding.create(
        input=chunk.text,
        model="text-embedding-ada-002"
    )['data'][0]['embedding']
    
    chunk.embedding = embedding
    
    # Store in chunk store
    key = (chunk.metadata.doc_id, chunk.metadata.chunk_id)
    chunk_store[key] = chunk
```

### 4. Perform Vector Search

```python
# Query your vector database (pseudo-code)
query = "What are the main types of machine learning?"
query_embedding = get_embedding(query)

# Get top-k results from your vector DB
vector_results = vector_db.search(query_embedding, k=6)

# Convert to SeedHits
seed_hits = [
    SeedHit(
        chunk=chunk_store[(result.doc_id, result.chunk_id)],
        similarity_score=result.score
    )
    for result in vector_results
]
```

### 5. Retrieve and Compose Spans

```python
# Configure retrieval
config = RetrievalConfig(
    window=2,                    # Expand ±2 neighbors
    k=6,                         # Use top-6 seeds
    token_budget=2000,           # Max 2000 tokens
    ordering_strategy="edge_balanced"
)

# Initialize composer
composer = SpanComposer(config)

# Retrieve spans
spans = composer.retrieve(
    seed_hits=seed_hits,
    chunk_store=chunk_store,
)
```

### 6. Use Retrieved Context

```python
# Build context for LLM
context = "\n\n".join(span.text for span in spans)

# Create prompt
prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {query}

Answer:"""

# Send to LLM
response = llm.complete(prompt)
```

## Chunking Your Documents

### Fixed-Size Chunking (Recommended)

Best for most use cases:

```python
from retrivex.utils import SimpleChunker

chunker = SimpleChunker(
    chunk_size=200,      # Target size in tokens
    overlap=50,          # Overlap for continuity
    chars_per_token=4,   # Rough conversion ratio
)

chunks = chunker.chunk_text(text, doc_id="doc1")
```

### Sentence-Based Chunking

For sentence-level precision:

```python
from retrivex.utils import SentenceChunker

chunker = SentenceChunker(
    sentences_per_chunk=3,    # 3 sentences per chunk
    overlap_sentences=1,      # 1 sentence overlap
)

chunks = chunker.chunk_text(text, doc_id="doc1")
```

### Custom Chunking

Implement your own chunker:

```python
def custom_chunk(text, doc_id):
    # Your custom logic
    chunks = []
    for i, section in enumerate(split_by_heading(text)):
        metadata = ChunkMetadata(
            doc_id=doc_id,
            chunk_id=i,
            char_start=section.start,
            char_end=section.end,
            heading_path=section.headings,
        )
        chunks.append(Chunk(text=section.text, metadata=metadata))
    return chunks
```

## Setting Up Vector Search

RetriVex works with any vector database. Here are examples:

### With FAISS

```python
import faiss
import numpy as np

# Create index
dimension = 1536  # for OpenAI embeddings
index = faiss.IndexFlatIP(dimension)  # Inner product

# Add vectors
embeddings = np.array([chunk.embedding for chunk in chunks])
index.add(embeddings)

# Search
query_vector = get_embedding(query)
scores, indices = index.search(np.array([query_vector]), k=6)

# Convert to SeedHits
seed_hits = [
    SeedHit(chunk=chunks[idx], similarity_score=scores[0][i])
    for i, idx in enumerate(indices[0])
]
```

### With ChromaDB

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

# Add chunks
collection.add(
    embeddings=[chunk.embedding for chunk in chunks],
    metadatas=[{
        "doc_id": chunk.metadata.doc_id,
        "chunk_id": chunk.metadata.chunk_id,
    } for chunk in chunks],
    ids=[f"{chunk.metadata.doc_id}_{chunk.metadata.chunk_id}" for chunk in chunks],
)

# Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=6,
)

# Convert to SeedHits (similar to above)
```

## Configuring RetriVex

### Basic Configuration

```python
config = RetrievalConfig(
    window=2,              # ±2 neighbors
    k=6,                   # Top-6 seeds
    token_budget=2000,     # Max tokens
)
```

### Advanced Configuration

```python
config = RetrievalConfig(
    # Core parameters
    window=3,                          # Aggressive expansion
    k=8,                               # More seeds
    token_budget=4000,                 # Higher budget
    
    # Scoring weights (must sum to reasonable values)
    similarity_weight=1.0,             # Base relevance
    adjacency_weight=0.8,              # Stronger neighbor preference
    continuity_weight=0.3,             # More continuity emphasis
    parent_weight=0.1,                 # Parent bonus
    
    # Adjacency decay
    distance_decay_beta=0.5,           # Slower decay (prefer distant neighbors)
    
    # Boundaries
    stop_on_heading_change=True,       # Stop at section boundaries
    min_continuity=0.6,                # Higher continuity requirement
    
    # Constraints
    max_spans=10,                      # Limit number of spans
    max_parent_chars=3000,             # Larger parent sections
    
    # Ordering
    ordering_strategy="edge_balanced", # Counter U-shaped bias
    
    # De-duplication
    dedupe_overlaps=True,              # Remove overlaps
    prefer_fewer_spans=True,           # Prefer longer spans
)
```

## Using Retrieved Spans

### Basic Usage

```python
# Get text
context = "\n\n".join(span.text for span in spans)
```

### With Provenance

```python
for i, span in enumerate(spans):
    print(f"Span {i+1}:")
    print(f"  From: {span.doc_id}")
    print(f"  Chunks: {span.chunk_ids}")
    print(f"  Score: {span.score:.3f}")
    print(f"  Text: {span.text[:100]}...")
```

### With Citations

```python
def format_with_citations(spans):
    context_parts = []
    for i, span in enumerate(spans):
        citation = f"[{i+1}]"
        text = f"{span.text} {citation}"
        context_parts.append(text)
    
    context = "\n\n".join(context_parts)
    
    # Add sources
    sources = "\n\nSources:\n" + "\n".join([
        f"[{i+1}] {span.doc_id}, chunks {span.chunk_ids[0]}-{span.chunk_ids[-1]}"
        for i, span in enumerate(spans)
    ])
    
    return context + sources
```

## Integration Examples

### With LangChain (Current - Manual)

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Your existing LangChain setup
vectorstore = FAISS.from_documents(documents, OpenAIEmbeddings())

# Get base results
base_results = vectorstore.similarity_search_with_score(query, k=6)

# Convert to RetriVex format
seed_hits = [
    SeedHit(
        chunk=convert_to_chunk(doc),
        similarity_score=score
    )
    for doc, score in base_results
]

# Use RetriVex
spans = composer.retrieve(seed_hits, chunk_store)
```

### With OpenAI

```python
from openai import OpenAI

client = OpenAI()

# Get context from RetriVex
context = "\n\n".join(span.text for span in spans)

# Create prompt
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "Answer based on the provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
    ]
)
```

## Performance Tuning

### For Precision (Reduce Noise)

```python
config = RetrievalConfig(
    window=1,                    # Less expansion
    similarity_weight=1.2,       # Emphasize relevance
    adjacency_weight=0.4,        # Less neighbor influence
    stop_on_heading_change=True, # Respect boundaries
    min_continuity=0.7,          # High continuity requirement
)
```

### For Recall (Get More Context)

```python
config = RetrievalConfig(
    window=3,                    # More expansion
    similarity_weight=0.8,       # Less strict on relevance
    adjacency_weight=0.8,        # Strong neighbor influence
    stop_on_heading_change=False,# Cross boundaries
    token_budget=4000,           # Larger budget
)
```

### For Speed (Minimize Overhead)

```python
config = RetrievalConfig(
    window=1,                    # Minimal expansion
    k=4,                         # Fewer seeds
    continuity_weight=0.0,       # Skip continuity computation
    dedupe_overlaps=False,       # Skip deduplication
)
```

### For Long Documents

```python
config = RetrievalConfig(
    window=2,
    stop_on_heading_change=True, # Important for structure
    min_continuity=0.6,
    max_spans=8,                 # Limit spans
    ordering_strategy="edge_balanced",
)
```

## Common Patterns

### Pattern 1: Multi-Hop QA

For questions requiring multiple pieces of information:

```python
config = RetrievalConfig(
    window=2,
    k=8,                         # More seeds for coverage
    token_budget=3000,
    ordering_strategy="edge_balanced",
)
```

### Pattern 2: Precise Extraction

For extracting specific facts:

```python
config = RetrievalConfig(
    window=1,
    similarity_weight=1.5,       # Emphasize exact match
    adjacency_weight=0.3,
    stop_on_heading_change=True,
)
```

### Pattern 3: Summarization

For generating summaries from retrieved context:

```python
config = RetrievalConfig(
    window=2,
    k=10,                        # Broad coverage
    token_budget=4000,           # More context
    ordering_strategy="score_desc",  # Best content first
)
```

### Pattern 4: Code Search

For retrieving code snippets:

```python
config = RetrievalConfig(
    window=3,                    # Include surrounding code
    stop_on_heading_change=True, # Respect function boundaries
    prefer_fewer_spans=True,     # Keep functions together
)
```

## Troubleshooting

### Issue: Too Much Irrelevant Content

**Solution**: Reduce window size, increase similarity_weight, enable stop_on_heading_change

```python
config = RetrievalConfig(
    window=1,
    similarity_weight=1.2,
    stop_on_heading_change=True,
    min_continuity=0.7,
)
```

### Issue: Missing Relevant Information

**Solution**: Increase window size, token budget, reduce continuity requirements

```python
config = RetrievalConfig(
    window=3,
    token_budget=4000,
    min_continuity=0.3,
    stop_on_heading_change=False,
)
```

### Issue: Slow Performance

**Solution**: Reduce window, disable continuity computation, use fewer seeds

```python
config = RetrievalConfig(
    window=1,
    k=4,
    continuity_weight=0.0,
)
```

### Issue: Fragmented Spans

**Solution**: Enable prefer_fewer_spans, increase adjacency_weight

```python
config = RetrievalConfig(
    prefer_fewer_spans=True,
    adjacency_weight=0.8,
)
```

## Next Steps

- Read [Concepts](concepts.md) to understand the theory
- Check [examples/quickstart.py](../examples/quickstart.py) for a complete example
- See [CONTRIBUTING.md](../CONTRIBUTING.md) to contribute
- Join discussions for questions and feature requests

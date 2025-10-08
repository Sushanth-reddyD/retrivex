"""
Quick Start Example for RetriVex

This example demonstrates the basic usage of RetriVex for neighbor-expansion retrieval.
"""

import hashlib

import numpy as np

from retrivex import (
    Chunk,
    ChunkMetadata,
    RetrievalConfig,
    SeedHit,
    SpanComposer,
    OrderingStrategy,
)
from retrivex.utils import SimpleChunker


def create_sample_document():
    """Create a sample document for demonstration."""
    return """
    Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence that focuses on developing
    algorithms that can learn from and make predictions or decisions based on data.
    
    Types of Machine Learning
    
    There are three main types of machine learning: supervised learning, unsupervised
    learning, and reinforcement learning. Each type has its own characteristics and
    use cases.
    
    Supervised Learning
    
    In supervised learning, the algorithm learns from labeled training data. The model
    is trained on input-output pairs and learns to map inputs to outputs. Common
    applications include classification and regression tasks.
    
    Unsupervised Learning
    
    Unsupervised learning works with unlabeled data. The algorithm tries to find
    patterns and structures in the data without explicit guidance. Clustering and
    dimensionality reduction are typical unsupervised learning tasks.
    
    Reinforcement Learning
    
    Reinforcement learning involves an agent learning to make decisions by interacting
    with an environment. The agent receives rewards or penalties based on its actions
    and learns to maximize cumulative reward over time.
    
    Applications
    
    Machine learning has numerous applications including image recognition, natural
    language processing, recommendation systems, autonomous vehicles, and medical
    diagnosis. The field continues to grow rapidly with new applications emerging.
    """


def create_simple_embeddings(text: str, dim: int = 10) -> list:
    """Create simple embeddings for demonstration (not production-ready)."""
    # Use deterministic SHA-256 hash for reproducibility across processes
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    np.random.seed(seed)
    embedding = np.random.rand(dim).tolist()
    return embedding


def main():
    """Demonstrate RetriVex usage."""

    print("=" * 70)
    print("RetriVex Quick Start Example")
    print("=" * 70)
    print()

    # Step 1: Chunk the document
    print("Step 1: Chunking the document")
    print("-" * 70)

    document = create_sample_document()
    chunker = SimpleChunker(chunk_size=50, overlap=10)
    chunks = chunker.chunk_text(
        text=document,
        doc_id="ml_intro",
        heading_path=["Machine Learning", "Introduction"],
    )

    print(f"Created {len(chunks)} chunks")
    print()

    # Step 2: Create chunk store with embeddings
    print("Step 2: Creating chunk store with embeddings")
    print("-" * 70)

    chunk_store = {}
    for chunk in chunks:
        # Add simple embeddings (in production, use real embeddings)
        chunk.embedding = create_simple_embeddings(chunk.text)
        key = (chunk.metadata.doc_id, chunk.metadata.chunk_id)
        chunk_store[key] = chunk

    print(f"Stored {len(chunk_store)} chunks with embeddings")
    print()

    # Step 3: Simulate vector search results (seed hits)
    print("Step 3: Simulating vector search results")
    print("-" * 70)

    # For demo, let's say we queried "What are the types of machine learning?"
    # and got hits on chunks discussing types
    query = "What are the types of machine learning?"

    # Simulate finding relevant chunks (in production, use real vector search)
    seed_hits = [
        SeedHit(chunk=chunks[3], similarity_score=0.92),  # Types section
        SeedHit(chunk=chunks[4], similarity_score=0.88),  # Supervised learning
        SeedHit(chunk=chunks[6], similarity_score=0.85),  # Unsupervised learning
    ]

    print(f"Query: '{query}'")
    print(f"Found {len(seed_hits)} seed hits:")
    for i, hit in enumerate(seed_hits, 1):
        preview = hit.chunk.text[:60].replace("\n", " ").strip()
        print(f"  {i}. Score: {hit.similarity_score:.2f} - {preview}...")
    print()

    # Step 4: Configure RetriVex
    print("Step 4: Configuring RetriVex")
    print("-" * 70)

    config = RetrievalConfig(
        window=2,  # Expand ±2 neighbors
        k=3,  # Use top-3 seeds
        token_budget=1000,  # Max 1000 tokens
        similarity_weight=1.0,
        adjacency_weight=0.6,
        continuity_weight=0.2,
        ordering_strategy=OrderingStrategy.EDGE_BALANCED,
    )

    print("Configuration:")
    print(f"  Window: ±{config.window} neighbors")
    print(f"  Token budget: {config.token_budget}")
    print(f"  Ordering: {config.ordering_strategy.value}")
    print()

    # Step 5: Retrieve and compose spans
    print("Step 5: Retrieving and composing spans")
    print("-" * 70)

    composer = SpanComposer(config)
    spans = composer.retrieve(
        seed_hits=seed_hits,
        chunk_store=chunk_store,
    )

    print(f"Retrieved {len(spans)} spans")
    print()

    # Step 6: Display results
    print("Step 6: Final context (edge-balanced ordering)")
    print("=" * 70)

    for i, span in enumerate(spans, 1):
        print(f"\nSpan {i} (Chunks {span.chunk_ids[0]}-{span.chunk_ids[-1]}):")
        print(
            f"  Score: {span.score:.3f} (sim: {span.sim_score:.2f}, "
            f"adj: {span.adjacency_score:.2f}, cont: {span.continuity_score:.2f})"
        )
        print(f"  Tokens: ~{span.token_count}")
        print(
            f"  Position: {i}/{len(spans)} ({'beginning' if i == 1 else 'end' if i == len(spans) else 'middle'})"
        )
        print(f"  Text preview: {span.text[:100].replace(chr(10), ' ').strip()}...")

    print()
    print("=" * 70)
    print("Summary")
    print("-" * 70)
    total_tokens = sum(span.token_count for span in spans)
    print(f"Total spans: {len(spans)}")
    print(f"Total tokens: ~{total_tokens} (budget: {config.token_budget})")
    print(f"Average span score: {sum(s.score for s in spans) / len(spans):.3f}")
    print()

    # Step 7: Show benefit of edge-balanced ordering
    print("Step 7: Why edge-balanced ordering matters")
    print("-" * 70)
    print("LLMs have U-shaped attention patterns:")
    print("  ✓ High attention at BEGINNING of context")
    print("  ✗ Low attention in MIDDLE of context")
    print("  ✓ High attention at END of context")
    print()
    print("RetriVex places high-value spans at edges where attention is high,")
    print("reducing the 'lost-in-the-middle' effect!")
    print()

    # Step 8: Compare with other strategies
    print("Step 8: Comparing ordering strategies")
    print("-" * 70)

    strategies = [
        OrderingStrategy.SCORE_DESC,
        OrderingStrategy.FRONT_FIRST,
        OrderingStrategy.BACK_FIRST,
    ]

    for strategy in strategies:
        config.ordering_strategy = strategy
        composer = SpanComposer(config)
        alt_spans = composer.retrieve(seed_hits=seed_hits, chunk_store=chunk_store)

        positions = [f"Chunk {s.chunk_ids[0]}" for s in alt_spans]
        print(f"{strategy.value:20s}: {' -> '.join(positions)}")

    print()
    print("=" * 70)
    print("Example complete! 🎉")
    print()
    print("Next steps:")
    print("  1. Use real embeddings (OpenAI, Cohere, HuggingFace)")
    print("  2. Integrate with your vector database")
    print("  3. Tune config parameters for your use case")
    print("  4. Evaluate on your specific tasks")
    print("=" * 70)


if __name__ == "__main__":
    main()

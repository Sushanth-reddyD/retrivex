"""
Advanced Example: Building a Q&A System with RetriVex

This example demonstrates a more realistic use case with multiple documents,
proper embeddings (simulated), and a complete Q&A pipeline.
"""

import hashlib

import numpy as np
from typing import Dict, List, Tuple

from retrivex import (
    Chunk,
    ChunkMetadata,
    OrderingStrategy,
    RetrievalConfig,
    SeedHit,
    SpanComposer,
)
from retrivex.utils import SimpleChunker


class SimpleVectorStore:
    """Simple in-memory vector store for demonstration."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: List[np.ndarray] = []

    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Add chunks to the store."""
        for chunk in chunks:
            if chunk.embedding is not None:
                self.chunks.append(chunk)
                self.embeddings.append(np.array(chunk.embedding))

    def search(self, query_embedding: List[float], k: int = 6) -> List[SeedHit]:
        """Search for similar chunks."""
        if not self.embeddings:
            return []

        query_vec = np.array(query_embedding)

        # Compute cosine similarities
        similarities = []
        for emb in self.embeddings:
            sim = np.dot(query_vec, emb) / (np.linalg.norm(query_vec) * np.linalg.norm(emb))
            similarities.append(float(sim))

        # Get top-k
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        # Create SeedHits
        return [
            SeedHit(chunk=self.chunks[idx], similarity_score=similarities[idx])
            for idx in top_k_indices
        ]


def create_sample_knowledge_base() -> Dict[str, str]:
    """Create a sample knowledge base with multiple documents."""
    return {
        "python_basics": """
        Python is a high-level, interpreted programming language known for its
        simplicity and readability. It was created by Guido van Rossum and first
        released in 1991.
        
        Variables and Data Types
        
        Python supports several data types including integers, floats, strings,
        lists, tuples, and dictionaries. Variables in Python are dynamically typed,
        meaning you don't need to declare their type explicitly.
        
        Control Flow
        
        Python provides standard control flow statements including if-else conditions,
        for loops, and while loops. The syntax uses indentation to define code blocks
        instead of curly braces.
        
        Functions
        
        Functions in Python are defined using the def keyword. They can accept
        parameters and return values. Python also supports lambda functions for
        simple one-line functions.
        """,
        "machine_learning": """
        Machine Learning Overview
        
        Machine learning is a subset of artificial intelligence that enables systems
        to learn and improve from experience without being explicitly programmed.
        
        Supervised Learning
        
        In supervised learning, models are trained on labeled data. Common algorithms
        include linear regression, logistic regression, decision trees, and neural
        networks. The goal is to learn a mapping from inputs to outputs.
        
        Unsupervised Learning
        
        Unsupervised learning works with unlabeled data to discover patterns. Common
        techniques include clustering (K-means, hierarchical), dimensionality reduction
        (PCA, t-SNE), and anomaly detection.
        
        Model Evaluation
        
        Machine learning models are evaluated using metrics like accuracy, precision,
        recall, F1-score, and ROC-AUC. Cross-validation is used to assess how well
        a model generalizes to unseen data.
        """,
        "web_development": """
        Web Development Fundamentals
        
        Web development involves building websites and web applications. It's typically
        divided into frontend (client-side) and backend (server-side) development.
        
        Frontend Technologies
        
        Frontend development uses HTML for structure, CSS for styling, and JavaScript
        for interactivity. Modern frameworks like React, Vue, and Angular make it
        easier to build complex user interfaces.
        
        Backend Technologies
        
        Backend development handles server logic, databases, and APIs. Popular backend
        languages include Python (Django, Flask), JavaScript (Node.js), Ruby (Rails),
        and Java (Spring). Databases can be SQL (PostgreSQL, MySQL) or NoSQL (MongoDB).
        
        RESTful APIs
        
        REST (Representational State Transfer) is an architectural style for APIs.
        RESTful APIs use HTTP methods (GET, POST, PUT, DELETE) and return data in
        formats like JSON or XML.
        """,
    }


def create_embedding(text: str, seed: int = 42) -> List[float]:
    """
    Create a simple embedding for demonstration.

    In production, use a real embedding model like OpenAI, Cohere, or HuggingFace.
    """
    # Use deterministic SHA-256 hash for reproducibility across processes
    # Mix in the seed to vary embeddings slightly
    combined = f"{text}_{seed}"
    hash_seed = int.from_bytes(hashlib.sha256(combined.encode("utf-8")).digest()[:4], "big")
    np.random.seed(hash_seed)
    embedding = np.random.rand(128).tolist()
    return embedding


def index_documents(
    documents: Dict[str, str], chunker: SimpleChunker
) -> Tuple[SimpleVectorStore, Dict[Tuple[str, int], Chunk]]:
    """Index documents into vector store and chunk store."""
    vector_store = SimpleVectorStore()
    chunk_store = {}

    print("Indexing documents...")
    for doc_id, text in documents.items():
        # Chunk the document
        chunks = chunker.chunk_text(text, doc_id=doc_id)

        # Create embeddings and store
        for chunk in chunks:
            chunk.embedding = create_embedding(chunk.text, seed=chunk.metadata.chunk_id)

            # Store in chunk store
            key = (chunk.metadata.doc_id, chunk.metadata.chunk_id)
            chunk_store[key] = chunk

        # Add to vector store
        vector_store.add_chunks(chunks)

        print(f"  Indexed {len(chunks)} chunks from '{doc_id}'")

    print(f"Total chunks indexed: {len(chunk_store)}\n")
    return vector_store, chunk_store


def answer_question(
    query: str,
    vector_store: SimpleVectorStore,
    chunk_store: Dict[Tuple[str, int], Chunk],
    composer: SpanComposer,
) -> Dict:
    """Answer a question using RetriVex."""
    print(f"Query: {query}")
    print("-" * 70)

    # Step 1: Vector search
    query_embedding = create_embedding(query)
    seed_hits = vector_store.search(query_embedding, k=6)

    print(f"Found {len(seed_hits)} seed hits:")
    for i, hit in enumerate(seed_hits[:3], 1):  # Show top 3
        preview = hit.chunk.text[:60].replace("\n", " ").strip()
        print(f"  {i}. {hit.chunk.metadata.doc_id} (score: {hit.similarity_score:.3f})")
        print(f"     {preview}...")

    # Step 2: Retrieve and compose spans
    spans = composer.retrieve(seed_hits=seed_hits, chunk_store=chunk_store)

    print(f"\nRetrieved {len(spans)} spans:")
    for i, span in enumerate(spans, 1):
        print(
            f"  {i}. {span.doc_id} chunks {span.chunk_ids[0]}-{span.chunk_ids[-1]} "
            f"(score: {span.score:.3f}, tokens: ~{span.token_count})"
        )

    # Step 3: Build context
    context = "\n\n".join([f"[{i+1}] {span.text}" for i, span in enumerate(spans)])

    total_tokens = sum(span.token_count for span in spans)
    print(f"\nTotal context: ~{total_tokens} tokens")

    return {
        "query": query,
        "context": context,
        "spans": spans,
        "seed_hits": seed_hits,
        "total_tokens": total_tokens,
    }


def main():
    """Run the advanced example."""
    print("=" * 70)
    print("Advanced RetriVex Example: Q&A System")
    print("=" * 70)
    print()

    # Step 1: Create knowledge base
    print("Step 1: Setting up knowledge base")
    print("-" * 70)
    documents = create_sample_knowledge_base()
    print(f"Created knowledge base with {len(documents)} documents")
    print()

    # Step 2: Initialize chunker
    print("Step 2: Initializing chunker")
    print("-" * 70)
    chunker = SimpleChunker(chunk_size=100, overlap=20)
    print(f"Chunker: {chunker.chunk_size} tokens, {chunker.overlap} overlap")
    print()

    # Step 3: Index documents
    print("Step 3: Indexing documents")
    print("-" * 70)
    vector_store, chunk_store = index_documents(documents, chunker)

    # Step 4: Configure RetriVex
    print("Step 4: Configuring RetriVex")
    print("-" * 70)
    config = RetrievalConfig(
        window=2,
        k=6,
        token_budget=800,
        similarity_weight=1.0,
        adjacency_weight=0.6,
        continuity_weight=0.2,
        ordering_strategy=OrderingStrategy.EDGE_BALANCED,
    )
    print(f"Config: window={config.window}, k={config.k}, budget={config.token_budget}")
    print()

    composer = SpanComposer(config)

    # Step 5: Answer questions
    print("Step 5: Answering questions")
    print("=" * 70)
    print()

    questions = [
        "What is supervised learning in machine learning?",
        "What are the main data types in Python?",
        "How do RESTful APIs work?",
    ]

    results = []
    for i, question in enumerate(questions, 1):
        print(f"\nQuestion {i}/{len(questions)}:")
        result = answer_question(question, vector_store, chunk_store, composer)
        results.append(result)
        print()
        print("=" * 70)

    # Step 6: Summary
    print("\nSummary")
    print("-" * 70)
    print(f"Total questions answered: {len(results)}")
    avg_tokens = sum(r["total_tokens"] for r in results) / len(results)
    print(f"Average context size: ~{avg_tokens:.0f} tokens")
    print()

    # Step 7: Show retrieval quality
    print("Retrieval Quality Analysis")
    print("-" * 70)
    for i, result in enumerate(results, 1):
        print(f"\nQuestion {i}: {result['query'][:50]}...")
        print(f"  Seed hits from documents:")
        doc_ids = set(hit.chunk.metadata.doc_id for hit in result["seed_hits"])
        for doc_id in doc_ids:
            count = sum(1 for hit in result["seed_hits"] if hit.chunk.metadata.doc_id == doc_id)
            print(f"    - {doc_id}: {count} hits")

        print(f"  Final spans from documents:")
        span_doc_ids = set(span.doc_id for span in result["spans"])
        for doc_id in span_doc_ids:
            span_count = sum(1 for span in result["spans"] if span.doc_id == doc_id)
            print(f"    - {doc_id}: {span_count} spans")

    print()
    print("=" * 70)
    print("Example complete!")
    print()
    print("Key Takeaways:")
    print("  1. RetriVex expanded seed hits to include relevant neighbors")
    print("  2. Edge-balanced ordering places high-value content at edges")
    print("  3. Token budget was respected across all queries")
    print("  4. Spans provide provenance for citations")
    print("=" * 70)


if __name__ == "__main__":
    main()

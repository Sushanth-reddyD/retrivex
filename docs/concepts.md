# Concepts: Understanding Positional Bias in LLMs

## The Lost-in-the-Middle Problem

Large Language Models (LLMs) exhibit a well-documented **U-shaped attention pattern** when processing long contexts. This means:

- ✅ **High attention** at the **beginning** of the context
- ❌ **Low attention** in the **middle** of the context  
- ✅ **High attention** at the **end** of the context

### Why This Matters for RAG

In Retrieval-Augmented Generation (RAG) systems, you typically:

1. Retrieve k relevant chunks via vector search
2. Concatenate them into a prompt
3. Send to LLM for answer generation

**The problem**: If you place all retrieved content in the middle of your prompt (between instruction and question), the LLM may miss crucial information even when it's present in the context.

### Research Evidence

Multiple studies have documented this phenomenon:

1. **[Lost in the Middle (Liu et al., 2023)](https://arxiv.org/abs/2307.03172)**
   - Showed that models perform poorly when relevant information is in the middle
   - Tested on multiple-document QA tasks
   - Found consistent U-shaped performance curves across model sizes

2. **[Found in the Middle (Liu et al., 2023)](https://arxiv.org/abs/2311.09198)**
   - Proposed calibration techniques to address positional bias
   - Demonstrated improved performance with position-aware prompting

3. **[LongBench (Bai et al., 2023)](https://arxiv.org/abs/2308.14508)**
   - Comprehensive benchmark for long-context understanding
   - Includes tasks where positional bias significantly impacts performance

## How RetriVex Addresses This

RetriVex tackles positional bias through four key mechanisms:

### 1. Neighbor Expansion

Instead of using only the top-k chunks from vector search, RetriVex expands each chunk to include ±L neighbors:

```
Original: [Chunk 5]
Expanded: [Chunk 3] [Chunk 4] [Chunk 5] [Chunk 6] [Chunk 7]
```

**Benefits**:
- Captures context that spans multiple chunks
- Reduces fragmentation
- Improves coherence

### 2. Intelligent Scoring

Each span is scored using multiple factors:

```python
score = w₀·similarity + w₁·adjacency + w₂·continuity + w₃·parent_bonus
```

- **Similarity** (w₀=1.0): Base relevance from vector search
- **Adjacency** (w₁=0.6): How close chunks are to seed hits (exponential decay)
- **Continuity** (w₂=0.2): Semantic flow between adjacent chunks
- **Parent Bonus** (w₃=0.1): Bonus for including parent sections

### 3. Edge-Balanced Ordering

RetriVex reorders spans to exploit the U-shaped attention pattern:

```
Traditional ordering:        Edge-balanced ordering:
[Instruction]                [Instruction]
[Span 1 - High value]        [Span 1 - High value]    ← Front (high attention)
[Span 2 - Medium value]      [Span 3 - Medium value]  ← Middle (low attention)  
[Span 3 - Low value]         [Span 2 - Medium value]  ← Back (high attention)
[Question]                   [Question]
```

### 4. Span Composition

Merges contiguous chunks into cohesive spans to reduce fragmentation:

```
Before: [Chunk 1] [Chunk 2] [Chunk 5] [Chunk 6] [Chunk 7]
After:  [Span A: Chunks 1-2] [Span B: Chunks 5-7]
```

## Expected Improvements

Based on the problem statement and research literature, you can expect:

### Primary Gains

- **+5-15% absolute improvement** on "adjacent-answer" cases
  - Cases where the answer spans multiple adjacent chunks
  - Situations where the most similar chunk is adjacent to the answer

- **≤10% regression** on non-adjacent cases
  - When the answer is contained in a single chunk
  - Minimal overhead from expansion

### E2E Metrics

- **Improved F1/EM** at fixed token budget
- **Better recall@k_span** (answer contained in merged spans)
- **Higher "rescued-by-neighbor %"** (neighbors contained the answer when seed didn't)

### Use Cases Where RetriVex Excels

1. **Multi-sentence answers**: Answer spans 2-3 sentences across chunks
2. **Procedural content**: Step-by-step instructions split across chunks
3. **Definitional content**: Term definition + explanation in adjacent chunks
4. **Long-form QA**: Questions requiring synthesis of nearby information

## When NOT to Use RetriVex

RetriVex may not help (or could hurt) when:

1. **Answers are always single-chunk**
   - Medical entity extraction
   - Single-fact lookup
   
2. **Documents lack structure**
   - Random social media posts
   - Unordered collections
   
3. **Very short contexts**
   - Already using <1k tokens total
   - No room for neighbor expansion

4. **Answers require distant chunks**
   - Information scattered across document
   - Better to use multiple retrieval rounds

## Comparison with Related Techniques

### Parent-Document Retrieval (LangChain)

**How it works**: Retrieve small chunks, return entire parent document

**Differences**:
- Parent-Doc: Hierarchical (chunk → parent)
- RetriVex: Horizontal (chunk → neighbors)
- Parent-Doc: May include too much irrelevant content
- RetriVex: Precise, token-budget aware

**When to use Parent-Doc**: Answers always at document-level granularity

### Sentence-Window Retrieval (LlamaIndex)

**How it works**: Retrieve sentences, include ±N surrounding sentences

**Differences**:
- Sentence-Window: Fixed window, no scoring
- RetriVex: Scored spans, position-aware ordering
- Sentence-Window: Simpler
- RetriVex: More sophisticated, handles U-shaped bias

**When to use Sentence-Window**: Very simple use case, single-sentence answers

### Auto-Merging (LlamaIndex)

**How it works**: Hierarchical merging of chunks based on relationships

**Differences**:
- Auto-Merging: Complex hierarchy management
- RetriVex: Lightweight neighbor expansion
- Auto-Merging: Better for complex document structures
- RetriVex: Faster, more transparent

**When to use Auto-Merging**: Complex hierarchical documents (legal, technical specs)

## Implementation Details

### Neighbor Expansion Algorithm

```python
for each seed_hit in top_k_hits:
    for delta in range(1, window + 1):
        # Check backward neighbor
        neighbor_id = chunk_id - delta
        if can_expand(neighbor_id):
            add_to_span(neighbor_id)
        
        # Check forward neighbor  
        neighbor_id = chunk_id + delta
        if can_expand(neighbor_id):
            add_to_span(neighbor_id)
```

### Continuity Checking

Prevents expansion across structural boundaries:

```python
if stop_on_heading_change:
    if seed_heading != neighbor_heading:
        if continuity(seed, neighbor) < min_continuity:
            # Don't expand - different sections
            break
```

### Adjacency Scoring

Exponential decay rewards closer neighbors:

```python
adjacency_score = Σ exp(-β × |chunk_id - seed_id|)
```

With β=0.7:
- Distance 0 (seed): 1.00
- Distance 1: 0.50
- Distance 2: 0.25
- Distance 3: 0.12

## Tuning Guidelines

### Window Size (L)

- **L=1**: Conservative, minimal expansion
- **L=2**: Default, good balance (recommended)
- **L=3**: Aggressive, use for very fragmented docs
- **L≥4**: Usually unnecessary, risks including noise

### Token Budget

- **500-1000**: Short-context models (GPT-3.5)
- **2000-4000**: Medium-context models (GPT-4)
- **8000+**: Long-context models (GPT-4-32k, Claude)

### Weights

**Default**: sim:1.0, adj:0.6, cont:0.2, parent:0.1

**Tune for**:
- More precision → Increase `similarity_weight`
- More recall → Increase `adjacency_weight`
- Better flow → Increase `continuity_weight`

### Ordering Strategy

- **edge_balanced** (default): General-purpose, counters U-shaped bias
- **score_desc**: Traditional, best-first
- **front_first**: Put everything at beginning (if model prefers start)
- **back_first**: Put everything at end (if model prefers recency)

## Evaluation Methodology

To properly evaluate RetriVex on your data:

1. **Create test sets**:
   - Adjacent-answer cases (answer spans chunks)
   - Single-chunk cases (control)
   - Position-varied cases (answer at different positions)

2. **Measure metrics**:
   - Recall@k_span: % where span contains answer
   - Rescued-by-neighbor %: Non-seed chunks had answer
   - F1/EM at fixed token budget
   - Latency overhead (should be <10ms)

3. **Run ablations**:
   - Base kNN only
   - +neighbors (no continuity)
   - +neighbors+continuity
   - Different ordering strategies

4. **Analyze by position**:
   - Plot performance vs. answer position (beginning/middle/end)
   - Verify edge-balanced improves middle performance

## Further Reading

1. **Positional Bias**:
   - [Lost in the Middle](https://arxiv.org/abs/2307.03172)
   - [Found in the Middle](https://arxiv.org/abs/2311.09198)

2. **Long-Context Evaluation**:
   - [LongBench](https://arxiv.org/abs/2308.14508)
   - [RULER](https://arxiv.org/abs/2404.06654)
   - [NIAH (Needle in a Haystack)](https://github.com/gkamradt/LLMTest_NeedleInAHaystack)

3. **Related Techniques**:
   - [LangChain Parent Document Retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever)
   - [LlamaIndex Sentence Window](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/MetadataReplacementDemo.html)

4. **Advanced Reranking**:
   - [ColBERT](https://arxiv.org/abs/2004.12832)
   - [ColBERTv2](https://arxiv.org/abs/2112.01488)

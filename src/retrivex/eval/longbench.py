"""
LongBench evaluation implementation.

This module implements evaluation against LongBench datasets, focusing on
multi-task long-context QA where answers often straddle chunk boundaries.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.models import Chunk, ChunkMetadata, SeedHit
from ..utils.chunking import SimpleChunker
from .base import BaseEvaluator, EvaluationConfig, EvaluationSample


class LongBenchEvaluator(BaseEvaluator):
    """
    Evaluator for LongBench datasets.

    LongBench focuses on long-context understanding with tasks like:
    - Multi-document QA
    - Long-form summarization
    - Code completion
    - Few-shot learning

    We focus on QA tasks where answers often span multiple chunks.
    """

    SUPPORTED_TASKS = [
        "narrativeqa",  # Story comprehension
        "qasper",  # Scientific paper QA
        "multifieldqa_en",  # Multi-domain QA
        "hotpotqa",  # Multi-hop reasoning
        "2wikimqa",  # Multi-hop QA with Wikipedia
        "gov_report",  # Government report summarization
        "qmsum",  # Meeting summarization
        "multi_news",  # Multi-document news summarization
        "vcsum",  # Video caption summarization
        "trec",  # Question classification
        "triviaqa",  # Trivia questions
        "samsum",  # Dialogue summarization
        "lsht",  # Long sequence understanding
        "passage_count",  # Passage counting
        "passage_retrieval_en",  # Passage retrieval
    ]

    def __init__(
        self,
        config: EvaluationConfig,
        data_dir: Optional[str] = None,
        tasks: Optional[List[str]] = None,
    ):
        super().__init__(config)
        self.data_dir = Path(data_dir) if data_dir else Path("data/longbench")
        self.tasks = tasks or ["narrativeqa", "qasper", "hotpotqa", "multifieldqa_en"]

        # Chunk configuration
        self.chunk_size = 500  # tokens
        self.chunk_overlap = 50  # tokens

        # Embedding simulation parameters
        self.embedding_dim = 384

    def load_dataset(self) -> List[EvaluationSample]:
        """Load LongBench dataset samples."""

        samples = []

        for task in self.tasks:
            if task not in self.SUPPORTED_TASKS:
                print(f"Warning: Task {task} not in supported tasks")
                continue

            task_samples = self._load_task_data(task)
            samples.extend(task_samples)

        # Shuffle for evaluation
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            random.shuffle(samples)

        return samples

    def _load_task_data(self, task: str) -> List[EvaluationSample]:
        """Load data for a specific LongBench task."""

        # Try to load from local data directory first
        task_file = self.data_dir / f"{task}.jsonl"

        if task_file.exists():
            return self._load_from_file(task_file, task)
        else:
            # Generate synthetic data that mimics LongBench characteristics
            print(f"Local data not found for {task}, generating synthetic samples")
            return self._generate_synthetic_task_data(task)

    def _load_from_file(self, file_path: Path, task: str) -> List[EvaluationSample]:
        """Load samples from JSONL file."""

        samples = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                try:
                    data = json.loads(line.strip())
                    sample = self._create_sample_from_data(data, task, line_idx)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line {line_idx} in {file_path}: {e}")
                    continue

        return samples

    def _create_sample_from_data(
        self, data: Dict[str, Any], task: str, sample_idx: int
    ) -> Optional[EvaluationSample]:
        """Create EvaluationSample from raw data."""

        try:
            # Extract fields based on LongBench format
            query = data.get("input", data.get("question", ""))
            context = data.get("context", data.get("passage", ""))
            answer = data.get("answers", data.get("answer", ""))

            if isinstance(answer, list):
                answer = answer[0] if answer else ""

            if not query or not context:
                return None

            # Create chunks
            chunks = self.create_chunks_from_text(context, f"{task}_{sample_idx}")

            # Find answer span in chunks (approximate)
            answer_chunk_ids = self._find_answer_chunks(answer, chunks)

            sample = EvaluationSample(
                sample_id=f"{task}_{sample_idx}",
                dataset="longbench",
                task_type=task,
                query=query,
                context=context,
                chunks=chunks,
                answer=answer,
                answer_chunk_ids=answer_chunk_ids,
                metadata={"task": task, "length": len(context), "num_chunks": len(chunks)},
            )

            return sample

        except Exception as e:
            print(f"Error creating sample from data: {e}")
            return None

    def _generate_synthetic_task_data(self, task: str) -> List[EvaluationSample]:
        """Generate synthetic data that mimics LongBench task characteristics."""

        samples = []
        num_samples = min(50, self.config.max_samples or 50)  # Limit synthetic samples

        for i in range(num_samples):
            if task == "narrativeqa":
                sample = self._generate_narrative_qa_sample(i)
            elif task == "qasper":
                sample = self._generate_scientific_qa_sample(i)
            elif task == "hotpotqa":
                sample = self._generate_multihop_qa_sample(i)
            elif task == "multifieldqa_en":
                sample = self._generate_multifield_qa_sample(i)
            else:
                sample = self._generate_generic_qa_sample(task, i)

            if sample:
                samples.append(sample)

        return samples

    def _generate_narrative_qa_sample(self, sample_idx: int) -> EvaluationSample:
        """Generate a narrative QA sample with answer spanning chunks."""

        # Create a story where the answer spans multiple paragraphs
        story_parts = [
            "The ancient castle stood on a hill overlooking the valley. Its stone walls had weathered countless storms, and its towers reached toward the cloudy sky. Legend spoke of a treasure hidden within its depths.",
            "Sir Edmund arrived at the castle gates on a rainy evening. The gatekeeper, an old man with a grey beard, welcomed him with suspicion. 'What brings you to this forgotten place?' he asked.",
            "Edmund explained his quest for the legendary Golden Crown of Valdor. The crown was said to possess magical properties that could heal any wound and grant wisdom to its wearer. Many knights had sought it before him.",
            "The gatekeeper's eyes gleamed with recognition. 'Ah, the Golden Crown,' he whispered. 'It lies in the eastern tower, but beware - the path is treacherous. Three challenges await: the riddle of the sphinx, the trial of courage, and the test of compassion.'",
            "Edmund thanked the gatekeeper and entered the castle. The corridors were dark and filled with echoes of the past. Ancient tapestries depicted the crown's history - how it was forged by the wizard Maltheus from starlight and blessed by the dragon of the north wind.",
            "As he climbed the spiral staircase to the eastern tower, Edmund pondered the challenges ahead. The crown's power was not just in its magic, but in the journey one took to claim it. Only those pure of heart could pass the trials.",
        ]

        context = "\n\n".join(story_parts)
        query = "What are the three challenges that await those seeking the Golden Crown of Valdor?"
        answer = "the riddle of the sphinx, the trial of courage, and the test of compassion"

        chunks = self.create_chunks_from_text(context, f"narrative_{sample_idx}")
        answer_chunk_ids = self._find_answer_chunks(answer, chunks)

        return EvaluationSample(
            sample_id=f"narrative_{sample_idx}",
            dataset="longbench",
            task_type="narrativeqa",
            query=query,
            context=context,
            chunks=chunks,
            answer=answer,
            answer_chunk_ids=answer_chunk_ids,
            metadata={"task": "narrativeqa", "synthetic": True},
        )

    def _generate_scientific_qa_sample(self, sample_idx: int) -> EvaluationSample:
        """Generate a scientific paper QA sample."""

        paper_sections = [
            "Abstract: This study investigates the effects of climate change on coral reef ecosystems in the Pacific Ocean. We analyzed data from 150 reef sites over a 20-year period to understand bleaching patterns and recovery rates.",
            "Introduction: Coral reefs are among the most biodiverse ecosystems on Earth, supporting approximately 25% of all marine species. However, rising ocean temperatures due to climate change pose a significant threat to these delicate environments.",
            "Methods: We collected temperature data, coral coverage measurements, and species diversity counts from reef sites across the Pacific. Statistical analysis was performed using ANOVA and regression models to identify trends and correlations.",
            "Results: Our findings show that reef sites in areas with temperature increases above 1.5°C experienced severe bleaching events. Recovery rates varied significantly, with northern reefs showing 40% recovery after 5 years, while southern reefs showed only 15% recovery.",
            "Discussion: The differential recovery rates suggest that northern reefs may have developed thermal tolerance through gradual exposure to warmer waters. This adaptation mechanism could be crucial for reef survival under continued climate change.",
            "Conclusion: Immediate action is needed to reduce greenhouse gas emissions and protect coral reefs. Conservation efforts should focus on northern reef populations as potential sources for reef restoration programs.",
        ]

        context = "\n\n".join(paper_sections)
        query = (
            "What were the recovery rates for northern and southern reefs after bleaching events?"
        )
        answer = "northern reefs showed 40% recovery after 5 years, while southern reefs showed only 15% recovery"

        chunks = self.create_chunks_from_text(context, f"scientific_{sample_idx}")
        answer_chunk_ids = self._find_answer_chunks(answer, chunks)

        return EvaluationSample(
            sample_id=f"scientific_{sample_idx}",
            dataset="longbench",
            task_type="qasper",
            query=query,
            context=context,
            chunks=chunks,
            answer=answer,
            answer_chunk_ids=answer_chunk_ids,
            metadata={"task": "qasper", "synthetic": True},
        )

    def _generate_multihop_qa_sample(self, sample_idx: int) -> EvaluationSample:
        """Generate a multi-hop reasoning QA sample."""

        knowledge_base = [
            "Marie Curie was a Polish-born physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the first person to win Nobel Prizes in two different scientific fields.",
            "The Nobel Prize in Physics 1903 was awarded jointly to Henri Becquerel, Pierre Curie, and Marie Curie for their work on radiation phenomena. This was the first Nobel Prize awarded to a woman.",
            "Marie Curie later won the Nobel Prize in Chemistry 1911 for the discovery of the elements radium and polonium. She named polonium after her homeland, Poland.",
            "Pierre Curie was Marie Curie's husband and research partner. Together, they discovered the elements polonium and radium. Tragically, Pierre Curie died in a street accident in 1906.",
            "After Pierre's death, Marie Curie continued their research and took over his teaching position at the University of Paris, becoming the first female professor in the university's history.",
            "The Curie Institute in Paris, founded by Marie Curie, remains one of the world's leading cancer research centers. Marie Curie died in 1934 from aplastic anemia, likely caused by prolonged exposure to radiation.",
        ]

        context = "\n\n".join(knowledge_base)
        query = "What element did Marie Curie name after her homeland, and what happened to her research partner?"
        answer = "Marie Curie named polonium after her homeland Poland, and her research partner Pierre Curie died in a street accident in 1906"

        chunks = self.create_chunks_from_text(context, f"multihop_{sample_idx}")
        answer_chunk_ids = self._find_answer_chunks(answer, chunks)

        return EvaluationSample(
            sample_id=f"multihop_{sample_idx}",
            dataset="longbench",
            task_type="hotpotqa",
            query=query,
            context=context,
            chunks=chunks,
            answer=answer,
            answer_chunk_ids=answer_chunk_ids,
            metadata={"task": "hotpotqa", "synthetic": True},
        )

    def _generate_multifield_qa_sample(self, sample_idx: int) -> EvaluationSample:
        """Generate a multi-field QA sample."""

        multi_domain_text = [
            "Technology: Artificial Intelligence has revolutionized many industries. Machine learning algorithms can now process vast amounts of data to identify patterns and make predictions with remarkable accuracy.",
            "Healthcare: AI applications in medicine include diagnostic imaging, drug discovery, and personalized treatment plans. Machine learning models can analyze medical images to detect diseases earlier than traditional methods.",
            "Finance: In the financial sector, AI is used for fraud detection, algorithmic trading, and risk assessment. Banks employ machine learning to analyze transaction patterns and identify suspicious activities.",
            "Transportation: Autonomous vehicles rely heavily on AI systems for navigation, obstacle detection, and decision-making. These systems process data from multiple sensors to ensure safe operation.",
            "Education: AI-powered educational platforms can provide personalized learning experiences, adapting to individual student needs and learning styles. This technology helps optimize educational outcomes.",
            "Environment: Climate scientists use AI to analyze weather patterns, predict natural disasters, and model climate change scenarios. Machine learning helps process vast amounts of environmental data.",
        ]

        context = "\n\n".join(multi_domain_text)
        query = "How is AI being used in healthcare and finance for detection purposes?"
        answer = "In healthcare, AI is used for diagnostic imaging to detect diseases earlier than traditional methods. In finance, AI is used for fraud detection by analyzing transaction patterns to identify suspicious activities."

        chunks = self.create_chunks_from_text(context, f"multifield_{sample_idx}")
        answer_chunk_ids = self._find_answer_chunks(answer, chunks)

        return EvaluationSample(
            sample_id=f"multifield_{sample_idx}",
            dataset="longbench",
            task_type="multifieldqa_en",
            query=query,
            context=context,
            chunks=chunks,
            answer=answer,
            answer_chunk_ids=answer_chunk_ids,
            metadata={"task": "multifieldqa_en", "synthetic": True},
        )

    def _generate_generic_qa_sample(self, task: str, sample_idx: int) -> EvaluationSample:
        """Generate a generic QA sample for other tasks."""

        # Generic long-form content
        content = [
            f"This is a {task} evaluation sample with multiple paragraphs of content.",
            "The content is designed to test retrieval across chunk boundaries.",
            "Each paragraph contains information that may be relevant to answering questions.",
            "The challenge is to identify and combine information from different sections.",
            "This simulates real-world scenarios where answers span multiple text segments.",
        ]

        context = "\n\n".join(content)
        query = f"What is the purpose of this {task} sample?"
        answer = "to test retrieval across chunk boundaries and combine information from different sections"

        chunks = self.create_chunks_from_text(context, f"{task}_{sample_idx}")
        answer_chunk_ids = self._find_answer_chunks(answer, chunks)

        return EvaluationSample(
            sample_id=f"{task}_{sample_idx}",
            dataset="longbench",
            task_type=task,
            query=query,
            context=context,
            chunks=chunks,
            answer=answer,
            answer_chunk_ids=answer_chunk_ids,
            metadata={"task": task, "synthetic": True},
        )

    def create_chunks(self, sample: EvaluationSample) -> List[Chunk]:
        """Create chunks from sample context."""
        return sample.chunks  # Already created during loading

    def create_chunks_from_text(self, text: str, doc_id: str) -> List[Chunk]:
        """Create chunks from raw text."""

        chunker = SimpleChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap)

        chunks = chunker.chunk_text(
            text=text, doc_id=doc_id, heading_path=["LongBench", "Document"]
        )

        # Add embeddings (simulated)
        for chunk in chunks:
            chunk.embedding = self._create_embedding(chunk.text)

        return chunks

    def _create_embedding(self, text: str) -> List[float]:
        """Create a simulated embedding for text."""
        # Simple hash-based embedding for demo
        np.random.seed(hash(text) % (2**32))
        embedding = np.random.rand(self.embedding_dim).tolist()
        return embedding

    def simulate_vector_search(self, query: str, chunks: List[Chunk], k: int = 6) -> List[SeedHit]:
        """Simulate vector search by computing text similarity."""

        query_embedding = self._create_embedding(query)

        # Compute similarities (using simple cosine similarity simulation)
        similarities = []
        for chunk in chunks:
            # Simulate similarity based on text overlap
            query_tokens = set(query.lower().split())
            chunk_tokens = set(chunk.text.lower().split())

            if not query_tokens or not chunk_tokens:
                similarity = 0.0
            else:
                intersection = query_tokens & chunk_tokens
                union = query_tokens | chunk_tokens
                similarity = len(intersection) / len(union)  # Jaccard similarity

            # Add some randomness to simulate embedding similarity
            np.random.seed(hash(chunk.text + query) % (2**32))
            noise = np.random.normal(0, 0.1)
            similarity = max(0.0, min(1.0, similarity + noise))

            similarities.append((similarity, chunk))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)

        seed_hits = [SeedHit(chunk=chunk, similarity_score=sim) for sim, chunk in similarities[:k]]

        return seed_hits

    def _find_answer_chunks(self, answer: str, chunks: List[Chunk]) -> List[int]:
        """Find which chunks contain parts of the answer."""

        answer_tokens = set(answer.lower().split())
        answer_chunk_ids = []

        for chunk in chunks:
            chunk_tokens = set(chunk.text.lower().split())

            # If chunk contains significant overlap with answer
            intersection = answer_tokens & chunk_tokens
            if len(intersection) >= max(1, len(answer_tokens) * 0.3):  # 30% overlap threshold
                answer_chunk_ids.append(chunk.metadata.chunk_id)

        return answer_chunk_ids

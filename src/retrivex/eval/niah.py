"""
Needle-in-a-Haystack (NIAH) evaluation implementation.

This module implements NIAH-style probes to quantify position effects quickly.
NIAH is designed to test whether models can find specific information placed
at different positions within long contexts.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.models import Chunk, ChunkMetadata, SeedHit
from ..utils.chunking import SimpleChunker
from .base import BaseEvaluator, EvaluationConfig, EvaluationSample


class NIAHEvaluator(BaseEvaluator):
    """
    Evaluator for Needle-in-a-Haystack style probes.

    NIAH tests measure the ability to find specific information (needles)
    placed at various positions within long, mostly irrelevant contexts (haystack).

    Key features:
    - Simple, focused needle insertion
    - Systematic position variation
    - Quick evaluation for position bias detection
    - Multiple needle types and difficulties
    - Variable haystack complexity
    """

    HAYSTACK_LENGTHS = [2000, 4000, 8000, 16000, 32000]  # Token counts
    NEEDLE_DEPTHS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Position ratios
    NEEDLE_DIFFICULTIES = ["easy", "medium", "hard"]

    def __init__(
        self,
        config: EvaluationConfig,
        haystack_lengths: Optional[List[int]] = None,
        needle_depths: Optional[List[float]] = None,
        needle_difficulties: Optional[List[str]] = None,
    ):
        super().__init__(config)

        self.haystack_lengths = haystack_lengths or [2000, 4000, 8000, 16000]
        self.needle_depths = needle_depths or [0.1, 0.3, 0.5, 0.7, 0.9]
        self.needle_difficulties = needle_difficulties or ["easy", "medium", "hard"]

        # Chunk configuration
        self.chunk_size = 300  # tokens
        self.chunk_overlap = 30  # tokens

        # Embedding simulation
        self.embedding_dim = 384

        # Haystack content pool
        self.haystack_pool = self._create_haystack_content_pool()

    def load_dataset(self) -> List[EvaluationSample]:
        """Generate NIAH evaluation samples with systematic position testing."""

        samples = []

        for haystack_length in self.haystack_lengths:
            for needle_depth in self.needle_depths:
                for difficulty in self.needle_difficulties:
                    # Generate multiple samples for each configuration
                    samples_per_config = max(
                        1,
                        min(
                            5,
                            (self.config.max_samples or 100)
                            // (
                                len(self.haystack_lengths)
                                * len(self.needle_depths)
                                * len(self.needle_difficulties)
                            ),
                        ),
                    )

                    for sample_idx in range(samples_per_config):
                        sample = self._generate_niah_sample(
                            haystack_length, needle_depth, difficulty, sample_idx
                        )
                        if sample:
                            samples.append(sample)

        # Shuffle for evaluation
        if self.config.random_seed:
            random.seed(self.config.random_seed)
            random.shuffle(samples)

        return samples

    def _generate_niah_sample(
        self, haystack_length: int, needle_depth: float, difficulty: str, sample_idx: int
    ) -> EvaluationSample:
        """Generate a single NIAH sample."""

        # Generate needle based on difficulty
        needle_info = self._generate_needle_by_difficulty(difficulty, sample_idx)
        needle_text = needle_info["text"]
        query = needle_info["query"]
        answer = needle_info["answer"]

        # Generate haystack content
        haystack_text = self._generate_haystack_content(
            haystack_length - len(needle_text.split()), difficulty
        )

        # Insert needle at specified depth
        full_context = self._insert_needle_at_depth(haystack_text, needle_text, needle_depth)

        # Create chunks
        doc_id = f"niah_{haystack_length}_{needle_depth:.1f}_{difficulty}_{sample_idx}"
        chunks = self.create_chunks_from_text(full_context, doc_id)

        # Find answer chunks
        answer_chunk_ids = self._find_answer_chunks(answer, chunks)

        # Calculate actual needle position
        actual_position = self._calculate_actual_position(needle_text, full_context)

        sample = EvaluationSample(
            sample_id=doc_id,
            dataset="niah",
            task_type=f"niah_{difficulty}",
            query=query,
            context=full_context,
            chunks=chunks,
            answer=answer,
            answer_chunk_ids=answer_chunk_ids,
            metadata={
                "haystack_length": haystack_length,
                "needle_depth": needle_depth,
                "difficulty": difficulty,
                "actual_position": actual_position,
                "needle_text": needle_text,
                "num_chunks": len(chunks),
                "needle_token_count": len(needle_text.split()),
            },
        )

        return sample

    def _generate_needle_by_difficulty(self, difficulty: str, sample_idx: int) -> Dict[str, str]:
        """Generate needle content based on difficulty level."""

        if difficulty == "easy":
            return self._generate_easy_needle(sample_idx)
        elif difficulty == "medium":
            return self._generate_medium_needle(sample_idx)
        elif difficulty == "hard":
            return self._generate_hard_needle(sample_idx)
        else:
            return self._generate_easy_needle(sample_idx)

    def _generate_easy_needle(self, sample_idx: int) -> Dict[str, str]:
        """Generate an easy-to-find needle with unique keywords."""

        needles = [
            {
                "text": "The magic spell to open the portal is 'ABRACADABRA MAXIMUS'.",
                "query": "What is the magic spell to open the portal?",
                "answer": "ABRACADABRA MAXIMUS",
            },
            {
                "text": "The treasure chest contains exactly 247 golden coins.",
                "query": "How many golden coins are in the treasure chest?",
                "answer": "247",
            },
            {
                "text": "Captain Blackbeard's ship is called the 'Crimson Revenge'.",
                "query": "What is the name of Captain Blackbeard's ship?",
                "answer": "Crimson Revenge",
            },
            {
                "text": "The secret ingredient in the potion is dragon's breath powder.",
                "query": "What is the secret ingredient in the potion?",
                "answer": "dragon's breath powder",
            },
            {
                "text": "The hidden door opens when you say 'SESAME MAGNIFICO'.",
                "query": "What phrase opens the hidden door?",
                "answer": "SESAME MAGNIFICO",
            },
        ]

        return needles[sample_idx % len(needles)]

    def _generate_medium_needle(self, sample_idx: int) -> Dict[str, str]:
        """Generate a medium difficulty needle with some ambiguity."""

        needles = [
            {
                "text": "Among all the documents, the most important file is stored in folder 'Project_Alpha_Final' under the name 'specifications_v3.pdf'.",
                "query": "What is the name of the most important file and where is it stored?",
                "answer": "specifications_v3.pdf in folder Project_Alpha_Final",
            },
            {
                "text": "The meeting scheduled for next Tuesday has been moved to Wednesday at 2:30 PM in conference room C-204.",
                "query": "When and where is the rescheduled meeting?",
                "answer": "Wednesday at 2:30 PM in conference room C-204",
            },
            {
                "text": "Dr. Sarah Johnson completed her research on renewable energy efficiency, achieving a breakthrough with 94.7% conversion rate using the new solar panel design.",
                "query": "What conversion rate did Dr. Sarah Johnson achieve with the new solar panel design?",
                "answer": "94.7%",
            },
            {
                "text": "The authentication process requires both fingerprint scan and the 12-digit code 847593621048 to access the secure database.",
                "query": "What is the 12-digit code required for authentication?",
                "answer": "847593621048",
            },
        ]

        return needles[sample_idx % len(needles)]

    def _generate_hard_needle(self, sample_idx: int) -> Dict[str, str]:
        """Generate a hard needle that requires careful attention."""

        needles = [
            {
                "text": "In the quarterly financial report, section 3.2.1 mentions that the subsidiary in Northern Europe recorded revenue of €2.3 million, while the footnote on page 47 clarifies that this figure excludes the December acquisition impact.",
                "query": "What revenue did the Northern Europe subsidiary record and what does this figure exclude?",
                "answer": "€2.3 million excluding December acquisition impact",
            },
            {
                "text": "The research protocol, as amended in version 4.2, requires participants to fast for exactly 14 hours before blood collection, with the exception that diabetic participants may consume up to 150ml of water with their medication.",
                "query": "How long must participants fast and what exception exists for diabetic participants?",
                "answer": "14 hours, diabetic participants may consume up to 150ml of water with medication",
            },
            {
                "text": "According to the compliance audit findings, department 7-G failed to implement the new safety procedures correctly, resulting in a temporary suspension of operations until the corrective action plan CAP-2024-077 is fully executed.",
                "query": "Which department failed compliance and what is the corrective action plan number?",
                "answer": "department 7-G, corrective action plan CAP-2024-077",
            },
        ]

        return needles[sample_idx % len(needles)]

    def _create_haystack_content_pool(self) -> List[str]:
        """Create a pool of haystack content paragraphs."""

        return [
            "The global economy continues to face unprecedented challenges as markets adapt to changing consumer behaviors and technological disruptions. Financial institutions are investing heavily in digital transformation initiatives to remain competitive in an increasingly connected world.",
            "Climate research indicates that atmospheric carbon dioxide levels have reached new highs, prompting international cooperation on emission reduction strategies. Renewable energy sources are becoming more cost-effective and reliable for large-scale deployment.",
            "Advances in artificial intelligence and machine learning are revolutionizing industries from healthcare to transportation. Companies are developing sophisticated algorithms to process vast amounts of data and extract meaningful insights for decision-making.",
            "Educational institutions worldwide are reimagining traditional learning models to incorporate digital technologies and personalized instruction methods. Online learning platforms have experienced unprecedented growth and adoption rates.",
            "The pharmaceutical industry continues to invest in research and development of new treatments for various diseases. Clinical trials are becoming more efficient through the use of advanced data analytics and patient recruitment technologies.",
            "Urban planning experts are designing smart cities that integrate sustainable infrastructure with technology-enabled services. These initiatives aim to improve quality of life while reducing environmental impact and resource consumption.",
            "Agricultural innovation focuses on precision farming techniques that optimize crop yields while minimizing environmental impact. Satellite imagery and IoT sensors provide farmers with real-time data for informed decision-making.",
            "Supply chain management has evolved to incorporate advanced forecasting models and automated inventory systems. Companies are building resilient networks that can adapt to disruptions and changing market demands.",
            "Cybersecurity remains a critical concern as organizations digitize their operations and store sensitive data in cloud environments. Security frameworks are constantly updated to address emerging threats and vulnerabilities.",
            "Healthcare technology continues to advance with the development of telemedicine platforms, wearable devices, and AI-assisted diagnostic tools. These innovations are improving patient outcomes and accessibility to medical care.",
            "Manufacturing processes are being transformed through automation, robotics, and predictive maintenance systems. Industry 4.0 initiatives are creating more efficient and flexible production environments.",
            "Environmental conservation efforts are leveraging satellite monitoring, drone surveillance, and data analytics to track ecosystem health and biodiversity. Conservation organizations are using technology to optimize their protection strategies.",
            "Financial technology companies are developing innovative payment solutions, digital banking services, and investment platforms. These fintech innovations are democratizing access to financial services globally.",
            "Transportation systems are evolving with electric vehicles, autonomous driving technology, and shared mobility solutions. Urban mobility is being reimagined to reduce congestion and environmental impact.",
            "Energy storage technology is advancing rapidly with improved battery systems and grid-scale solutions. These developments are essential for the widespread adoption of renewable energy sources.",
            "Biotechnology research is exploring gene therapy, personalized medicine, and advanced drug delivery systems. These innovations promise to revolutionize treatment approaches for various medical conditions.",
            "Space exploration continues to advance with private companies joining government agencies in developing new spacecraft and satellite technologies. Commercial space ventures are opening new opportunities for research and development.",
            "Materials science research is creating stronger, lighter, and more sustainable materials for construction, electronics, and consumer products. Nanotechnology applications are enabling breakthrough innovations across industries.",
            "Water management systems are incorporating smart sensors and IoT technology to optimize distribution and conservation efforts. These systems help address growing concerns about water scarcity in many regions.",
            "Waste management innovation focuses on recycling technologies, circular economy principles, and waste-to-energy solutions. Cities are implementing comprehensive strategies to reduce landfill dependency.",
        ]

    def _generate_haystack_content(self, target_length: int, difficulty: str) -> str:
        """Generate haystack content of specified length."""

        content = []
        current_length = 0
        used_paragraphs = set()

        while current_length < target_length:
            # Select a paragraph we haven't used yet
            available_paragraphs = [
                p for i, p in enumerate(self.haystack_pool) if i not in used_paragraphs
            ]
            if not available_paragraphs:
                # Reset if we've used all paragraphs
                used_paragraphs.clear()
                available_paragraphs = self.haystack_pool

            paragraph = random.choice(available_paragraphs)
            paragraph_idx = self.haystack_pool.index(paragraph)
            used_paragraphs.add(paragraph_idx)

            # Add some variation to the paragraph if needed
            if difficulty == "hard":
                paragraph = self._add_distractor_content(paragraph)

            content.append(paragraph)
            current_length += len(paragraph.split())

        return "\n\n".join(content)

    def _add_distractor_content(self, paragraph: str) -> str:
        """Add distractor content to make needle finding harder."""

        distractors = [
            "According to recent studies, this approach has shown promising results.",
            "Industry experts believe this trend will continue to evolve.",
            "Implementation requires careful consideration of various factors.",
            "Stakeholders are monitoring developments closely.",
            "Further research is needed to validate these findings.",
        ]

        # Randomly add a distractor sentence
        if random.random() < 0.3:  # 30% chance
            distractor = random.choice(distractors)
            sentences = paragraph.split(". ")
            insert_pos = random.randint(0, len(sentences))
            sentences.insert(insert_pos, distractor)
            paragraph = ". ".join(sentences)

        return paragraph

    def _insert_needle_at_depth(self, haystack_text: str, needle_text: str, depth: float) -> str:
        """Insert needle at specified depth in haystack."""

        paragraphs = haystack_text.split("\n\n")

        # Calculate insertion position based on depth
        insert_pos = int(len(paragraphs) * depth)
        insert_pos = max(0, min(insert_pos, len(paragraphs)))

        # Insert needle
        paragraphs.insert(insert_pos, needle_text)

        return "\n\n".join(paragraphs)

    def _calculate_actual_position(self, needle_text: str, full_context: str) -> float:
        """Calculate the actual character position ratio of the needle."""

        needle_start = full_context.find(needle_text)
        if needle_start == -1:
            return 0.5  # Default if not found

        return needle_start / len(full_context)

    def create_chunks(self, sample: EvaluationSample) -> List[Chunk]:
        """Create chunks from sample context."""
        return sample.chunks  # Already created during loading

    def create_chunks_from_text(self, text: str, doc_id: str) -> List[Chunk]:
        """Create chunks from raw text."""

        chunker = SimpleChunker(chunk_size=self.chunk_size, overlap=self.chunk_overlap)

        chunks = chunker.chunk_text(text=text, doc_id=doc_id, heading_path=["NIAH", "Document"])

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

        # Compute similarities
        similarities = []
        for chunk in chunks:
            # Simple text-based similarity
            query_tokens = set(query.lower().split())
            chunk_tokens = set(chunk.text.lower().split())

            if not query_tokens or not chunk_tokens:
                similarity = 0.0
            else:
                # Use Jaccard similarity as base
                intersection = query_tokens & chunk_tokens
                union = query_tokens | chunk_tokens
                similarity = len(intersection) / len(union)

                # Boost similarity if chunk contains exact answer phrases
                query_words = query.lower().split()
                chunk_text = chunk.text.lower()
                for i in range(len(query_words) - 1):
                    bigram = " ".join(query_words[i : i + 2])
                    if bigram in chunk_text:
                        similarity += 0.2  # Boost for bigram match

            # Add controlled randomness
            np.random.seed(hash(chunk.text + query) % (2**32))
            noise = np.random.normal(0, 0.03)  # Small noise
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
            chunk_text = chunk.text.lower()

            # Check for exact answer phrase
            if answer.lower() in chunk_text:
                answer_chunk_ids.append(chunk.metadata.chunk_id)
                continue

            # Check for token overlap
            chunk_tokens = set(chunk_text.split())
            intersection = answer_tokens & chunk_tokens

            # Lower threshold for NIAH since answers should be more contained
            if len(intersection) >= max(1, len(answer_tokens) * 0.5):  # 50% overlap threshold
                answer_chunk_ids.append(chunk.metadata.chunk_id)

        return answer_chunk_ids

    def analyze_depth_sensitivity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how performance varies with needle depth."""

        depth_analysis = {}

        for depth in self.needle_depths:
            depth_analysis[f"depth_{depth:.1f}"] = {
                "samples": 0,
                "avg_recall": 0.0,
                "avg_f1": 0.0,
                "avg_mrr": 0.0,
                "rescue_rate": 0.0,
            }

        # This would analyze actual results - placeholder for now
        return {
            "worst_depth": 0.5,  # Middle is typically worst
            "best_depth": 0.1,  # Beginning is typically best
            "performance_variance": 0.25,
            "u_shaped_pattern_detected": True,
            "depth_sensitivity_score": 0.3,  # Higher means more sensitive to position
        }

    def quick_position_probe(
        self, haystack_length: int = 8000, num_positions: int = 10
    ) -> List[EvaluationSample]:
        """Generate a quick position sensitivity probe."""

        samples = []
        depths = np.linspace(0.1, 0.9, num_positions)

        for i, depth in enumerate(depths):
            sample = self._generate_niah_sample(haystack_length, depth, "easy", i)
            if sample:
                samples.append(sample)

        return samples

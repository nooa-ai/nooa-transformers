"""
Hallucination Detection Benchmark for Grammatical Transformers

Based on Microsoft Research's Glass framework, measures grammatical symmetry
between input and output to detect and quantify hallucinations.

Hallucination Types:
1. Entity hallucination: Introducing entities not in source
2. Predicate hallucination: Changing or inventing actions
3. Negation errors: Flipping affirmative ↔ negative
4. Structural inconsistency: Breaking grammatical relationships

Theory: Hallucinations are grammatical inconsistencies. High symmetry → Low hallucinations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json

try:
    from transformers import AutoTokenizer
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

from ..models.grammatical_bert import (
    GrammaticalBertConfig,
    GrammaticalBertModel,
    GrammaticalBertForSequenceClassification
)
from ..chomsky.symmetry import GrammaticalSymmetry, SymmetryMetrics


@dataclass
class HallucinationExample:
    """Single hallucination test example"""
    source: str
    generated: str
    expected_symmetry: float
    hallucination_type: str
    should_detect: bool


class HallucinationBenchmark:
    """
    Comprehensive hallucination detection benchmark

    Measures:
    1. Entity consistency: Are source entities preserved?
    2. Predicate consistency: Are source predicates preserved?
    3. Negation preservation: Is negation handled correctly?
    4. Structural similarity: Are grammatical structures similar?

    Usage:
        benchmark = HallucinationBenchmark(
            model_name_or_path='bert-base-uncased',
            constituency_penalty=0.5
        )
        results = benchmark.run_benchmark()
    """

    def __init__(
        self,
        model_name_or_path: str = 'bert-base-uncased',
        constituency_penalty: float = 0.5,
        use_symmetry_loss: bool = True,
        symmetry_loss_weight: float = 0.1,
        device: str = 'cpu',
    ):
        """
        Args:
            model_name_or_path: Base model to load
            constituency_penalty: Cross-constituent attention penalty
            use_symmetry_loss: Whether to use symmetry loss
            symmetry_loss_weight: Weight for symmetry loss
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name_or_path = model_name_or_path
        self.constituency_penalty = constituency_penalty
        self.use_symmetry_loss = use_symmetry_loss
        self.symmetry_loss_weight = symmetry_loss_weight
        self.device = device

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # Load model
        self.model = self._create_model()
        self.model.to(device)
        self.model.eval()

        # Load spaCy for linguistic analysis
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
        else:
            self.nlp = None

        # Symmetry calculator
        self.symmetry_calculator = GrammaticalSymmetry()

    def _create_model(self) -> GrammaticalBertModel:
        """Create Grammatical BERT model"""
        config = GrammaticalBertConfig.from_pretrained(
            self.model_name_or_path,
            constituency_penalty=self.constituency_penalty,
            use_symmetry_loss=self.use_symmetry_loss,
            symmetry_loss_weight=self.symmetry_loss_weight,
        )
        model = GrammaticalBertModel(config)
        return model

    def extract_entities(self, text: str) -> Set[str]:
        """Extract entities from text using spaCy"""
        if self.nlp is None:
            # Fallback: simple noun extraction
            doc = text.lower().split()
            return set(doc)

        doc = self.nlp(text)
        entities = set()

        # Named entities
        for ent in doc.ents:
            entities.add(ent.text.lower())

        # Noun chunks
        for chunk in doc.noun_chunks:
            entities.add(chunk.text.lower())

        return entities

    def extract_predicates(self, text: str) -> Set[str]:
        """Extract predicates (verbs) from text"""
        if self.nlp is None:
            # Fallback: return empty set
            return set()

        doc = self.nlp(text)
        predicates = set()

        for token in doc:
            if token.pos_ == 'VERB':
                predicates.add(token.lemma_.lower())

        return predicates

    def has_negation(self, text: str) -> bool:
        """Check if text contains negation"""
        negation_words = {'not', 'no', 'never', 'nothing', 'nobody', 'nowhere', "n't", 'none'}

        if self.nlp is None:
            # Fallback: simple word matching
            words = text.lower().split()
            return any(word in negation_words for word in words)

        doc = self.nlp(text)
        for token in doc:
            if token.dep_ == 'neg' or token.text.lower() in negation_words:
                return True
        return False

    def compute_entity_consistency(self, source: str, generated: str) -> float:
        """
        Compute entity consistency

        Score = |entities_in_both| / |entities_in_source|
        1.0 = all source entities preserved
        0.0 = no source entities preserved
        """
        source_entities = self.extract_entities(source)
        generated_entities = self.extract_entities(generated)

        if len(source_entities) == 0:
            return 1.0

        overlap = source_entities & generated_entities
        consistency = len(overlap) / len(source_entities)

        return consistency

    def compute_predicate_consistency(self, source: str, generated: str) -> float:
        """
        Compute predicate (verb) consistency

        Score = |predicates_in_both| / |predicates_in_source|
        """
        source_predicates = self.extract_predicates(source)
        generated_predicates = self.extract_predicates(generated)

        if len(source_predicates) == 0:
            return 1.0

        overlap = source_predicates & generated_predicates
        consistency = len(overlap) / len(source_predicates)

        return consistency

    def compute_negation_consistency(self, source: str, generated: str) -> float:
        """
        Compute negation consistency

        1.0 = negation preserved (both have or both don't have)
        0.0 = negation flipped
        """
        source_negated = self.has_negation(source)
        generated_negated = self.has_negation(generated)

        # Both have negation or both don't have negation
        if source_negated == generated_negated:
            return 1.0
        else:
            return 0.0

    def compute_structural_similarity(self, source: str, generated: str) -> float:
        """
        Compute structural similarity using model embeddings

        Uses cosine similarity of pooled embeddings
        """
        # Tokenize
        source_encoded = self.tokenizer(
            source,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        generated_encoded = self.tokenizer(
            generated,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)

        # Get embeddings
        with torch.no_grad():
            source_output = self.model(**source_encoded)
            generated_output = self.model(**generated_encoded)

            source_embedding = source_output.pooler_output
            generated_embedding = generated_output.pooler_output

        # Cosine similarity
        similarity = nn.functional.cosine_similarity(
            source_embedding,
            generated_embedding,
            dim=1
        )

        return similarity.item()

    def compute_symmetry(
        self,
        source: str,
        generated: str,
        weights: Optional[Tuple[float, float, float, float]] = None
    ) -> SymmetryMetrics:
        """
        Compute complete grammatical symmetry

        Args:
            source: Source text
            generated: Generated text
            weights: (α, β, γ, δ) for entity, predicate, negation, structure

        Returns:
            SymmetryMetrics with individual scores and total
        """
        if weights is None:
            weights = (0.3, 0.3, 0.2, 0.2)  # Default weights

        α, β, γ, δ = weights

        # Compute individual metrics
        entity_score = self.compute_entity_consistency(source, generated)
        predicate_score = self.compute_predicate_consistency(source, generated)
        negation_score = self.compute_negation_consistency(source, generated)
        structural_score = self.compute_structural_similarity(source, generated)

        # Weighted combination
        total_symmetry = (
            α * entity_score +
            β * predicate_score +
            γ * negation_score +
            δ * structural_score
        )

        return SymmetryMetrics(
            entity_overlap=entity_score,
            predicate_overlap=predicate_score,
            negation_consistency=negation_score,
            structural_similarity=structural_score,
            total_score=total_symmetry
        )

    def evaluate_example(self, example: HallucinationExample) -> Dict:
        """Evaluate a single hallucination example"""
        symmetry = self.compute_symmetry(example.source, example.generated)

        # Hallucination detected if symmetry < threshold
        threshold = 0.7
        detected_hallucination = symmetry.total_score < threshold

        correct_detection = detected_hallucination == example.should_detect

        return {
            'source': example.source,
            'generated': example.generated,
            'symmetry': symmetry,
            'detected_hallucination': detected_hallucination,
            'should_detect': example.should_detect,
            'correct_detection': correct_detection,
            'hallucination_type': example.hallucination_type,
        }

    def get_test_examples(self) -> List[HallucinationExample]:
        """
        Get test examples for hallucination detection

        Returns examples with known hallucinations and correct outputs
        """
        return [
            # Entity hallucinations
            HallucinationExample(
                source="The dog barked loudly.",
                generated="The cat meowed softly.",
                expected_symmetry=0.3,
                hallucination_type="entity_substitution",
                should_detect=True
            ),
            HallucinationExample(
                source="John loves Mary.",
                generated="John loves Mary and Sarah.",
                expected_symmetry=0.7,
                hallucination_type="entity_addition",
                should_detect=True
            ),
            HallucinationExample(
                source="The quick brown fox jumps.",
                generated="The fast brown fox leaps.",
                expected_symmetry=0.85,
                hallucination_type="entity_synonym",
                should_detect=False  # Synonyms are acceptable
            ),

            # Predicate hallucinations
            HallucinationExample(
                source="The company announced earnings.",
                generated="The company hid earnings.",
                expected_symmetry=0.4,
                hallucination_type="predicate_substitution",
                should_detect=True
            ),
            HallucinationExample(
                source="She walked to school.",
                generated="She drove and walked to school.",
                expected_symmetry=0.6,
                hallucination_type="predicate_addition",
                should_detect=True
            ),

            # Negation errors
            HallucinationExample(
                source="The experiment was successful.",
                generated="The experiment was not successful.",
                expected_symmetry=0.2,
                hallucination_type="negation_flip",
                should_detect=True
            ),
            HallucinationExample(
                source="He did not go to the store.",
                generated="He went to the store.",
                expected_symmetry=0.3,
                hallucination_type="negation_removal",
                should_detect=True
            ),

            # Correct preservations (no hallucination)
            HallucinationExample(
                source="The weather is nice today.",
                generated="The weather is nice today.",
                expected_symmetry=1.0,
                hallucination_type="exact_match",
                should_detect=False
            ),
            HallucinationExample(
                source="AI systems can understand language.",
                generated="Artificial intelligence systems can comprehend language.",
                expected_symmetry=0.9,
                hallucination_type="paraphrase",
                should_detect=False
            ),

            # Structural inconsistencies
            HallucinationExample(
                source="The boy who was tall played basketball.",
                generated="The boy played. He was tall.",
                expected_symmetry=0.75,
                hallucination_type="structure_change",
                should_detect=False  # Structure change but semantics preserved
            ),
            HallucinationExample(
                source="Because it rained, the game was cancelled.",
                generated="The game was cancelled. It rained.",
                expected_symmetry=0.8,
                hallucination_type="causality_weakened",
                should_detect=False  # Causal relationship weakened but facts preserved
            ),
        ]

    def run_benchmark(self) -> Dict:
        """
        Run complete hallucination detection benchmark

        Returns:
            Dictionary with results and metrics
        """
        print("\n" + "="*60)
        print("Hallucination Detection Benchmark")
        print("="*60)
        print(f"Model: {self.model_name_or_path}")
        print(f"Constituency Penalty: {self.constituency_penalty}")
        print("="*60 + "\n")

        examples = self.get_test_examples()
        results = []

        print(f"Testing {len(examples)} examples...\n")

        for i, example in enumerate(examples, 1):
            print(f"Example {i}/{len(examples)}: {example.hallucination_type}")
            result = self.evaluate_example(example)
            results.append(result)

        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(results)

        # Print summary
        self._print_summary(results, metrics)

        return {
            'results': results,
            'metrics': metrics,
        }

    def _compute_aggregate_metrics(self, results: List[Dict]) -> Dict:
        """Compute aggregate metrics from results"""
        total = len(results)

        # Detection accuracy
        correct_detections = sum(1 for r in results if r['correct_detection'])
        detection_accuracy = correct_detections / total

        # Average symmetry scores
        avg_entity = np.mean([r['symmetry'].entity_overlap for r in results])
        avg_predicate = np.mean([r['symmetry'].predicate_overlap for r in results])
        avg_negation = np.mean([r['symmetry'].negation_consistency for r in results])
        avg_structural = np.mean([r['symmetry'].structural_similarity for r in results])
        avg_total = np.mean([r['symmetry'].total_score for r in results])

        # Separate metrics for hallucinated vs. correct
        hallucinated = [r for r in results if r['should_detect']]
        correct = [r for r in results if not r['should_detect']]

        avg_symmetry_hallucinated = np.mean([r['symmetry'].total_score for r in hallucinated])
        avg_symmetry_correct = np.mean([r['symmetry'].total_score for r in correct])

        # Hallucination rate (false positives + false negatives)
        false_positives = sum(1 for r in results if r['detected_hallucination'] and not r['should_detect'])
        false_negatives = sum(1 for r in results if not r['detected_hallucination'] and r['should_detect'])
        hallucination_rate = (false_positives + false_negatives) / total

        return {
            'detection_accuracy': detection_accuracy,
            'hallucination_rate': hallucination_rate,
            'avg_entity_consistency': avg_entity,
            'avg_predicate_consistency': avg_predicate,
            'avg_negation_consistency': avg_negation,
            'avg_structural_similarity': avg_structural,
            'avg_total_symmetry': avg_total,
            'avg_symmetry_hallucinated': avg_symmetry_hallucinated,
            'avg_symmetry_correct': avg_symmetry_correct,
            'symmetry_separation': avg_symmetry_correct - avg_symmetry_hallucinated,
            'total_examples': total,
            'correct_detections': correct_detections,
        }

    def _print_summary(self, results: List[Dict], metrics: Dict):
        """Print summary of benchmark results"""
        print("\n" + "="*60)
        print("Hallucination Detection Results")
        print("="*60 + "\n")

        print("Detection Metrics:")
        print(f"  Accuracy:           {metrics['detection_accuracy']:.2%}")
        print(f"  Hallucination Rate: {metrics['hallucination_rate']:.2%}")
        print(f"  Correct Detections: {metrics['correct_detections']}/{metrics['total_examples']}")

        print("\nSymmetry Metrics:")
        print(f"  Entity Consistency:     {metrics['avg_entity_consistency']:.4f}")
        print(f"  Predicate Consistency:  {metrics['avg_predicate_consistency']:.4f}")
        print(f"  Negation Consistency:   {metrics['avg_negation_consistency']:.4f}")
        print(f"  Structural Similarity:  {metrics['avg_structural_similarity']:.4f}")
        print(f"  Total Symmetry:         {metrics['avg_total_symmetry']:.4f}")

        print("\nSymmetry by Category:")
        print(f"  Hallucinated Examples:  {metrics['avg_symmetry_hallucinated']:.4f}")
        print(f"  Correct Examples:       {metrics['avg_symmetry_correct']:.4f}")
        print(f"  Separation:             {metrics['symmetry_separation']:.4f}")

        print("\n" + "="*60)

        # Detailed results by type
        print("\nResults by Hallucination Type:")
        print("="*60)

        type_results = defaultdict(list)
        for r in results:
            type_results[r['hallucination_type']].append(r)

        for halluc_type, type_results_list in sorted(type_results.items()):
            avg_symmetry = np.mean([r['symmetry'].total_score for r in type_results_list])
            accuracy = np.mean([r['correct_detection'] for r in type_results_list])
            print(f"{halluc_type:25} Symmetry: {avg_symmetry:.3f}  Accuracy: {accuracy:.0%}")

        print("="*60 + "\n")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Run hallucination benchmark on Grammatical BERT')
    parser.add_argument('--model', default='bert-base-uncased', help='Base model')
    parser.add_argument('--constituency_penalty', type=float, default=0.5, help='Constituency penalty')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    parser.add_argument('--output', default='hallucination_results.json', help='Output file')

    args = parser.parse_args()

    # Create benchmark
    benchmark = HallucinationBenchmark(
        model_name_or_path=args.model,
        constituency_penalty=args.constituency_penalty,
        device=args.device,
    )

    # Run
    results = benchmark.run_benchmark()

    # Save results
    with open(args.output, 'w') as f:
        # Convert SymmetryMetrics to dict for JSON serialization
        serializable_results = []
        for r in results['results']:
            serializable_r = dict(r)
            serializable_r['symmetry'] = {
                'entity_overlap': r['symmetry'].entity_overlap,
                'predicate_overlap': r['symmetry'].predicate_overlap,
                'negation_consistency': r['symmetry'].negation_consistency,
                'structural_similarity': r['symmetry'].structural_similarity,
                'total_score': r['symmetry'].total_score,
            }
            serializable_results.append(serializable_r)

        json.dump({
            'results': serializable_results,
            'metrics': results['metrics'],
        }, f, indent=2)

    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()

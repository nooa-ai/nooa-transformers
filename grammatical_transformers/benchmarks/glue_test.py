"""
GLUE Benchmark for Grammatical Transformers

Tests Grammatical BERT on the General Language Understanding Evaluation (GLUE) benchmark,
comparing performance with vanilla BERT baseline.

GLUE Tasks:
- MNLI: Multi-Genre Natural Language Inference
- QQP: Quora Question Pairs
- QNLI: Question Natural Language Inference
- SST-2: Stanford Sentiment Treebank
- CoLA: Corpus of Linguistic Acceptability
- STS-B: Semantic Textual Similarity Benchmark
- MRPC: Microsoft Research Paraphrase Corpus
- RTE: Recognizing Textual Entailment
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

try:
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        Trainer,
        TrainingArguments,
        EvalPrediction
    )
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from ..models.grammatical_bert import (
    GrammaticalBertConfig,
    GrammaticalBertForSequenceClassification
)


@dataclass
class GLUETaskConfig:
    """Configuration for a GLUE task"""
    name: str
    dataset_name: str
    text_fields: Tuple[str, ...]
    num_labels: int
    metric: str
    is_regression: bool = False


# GLUE task configurations
GLUE_TASKS = {
    'mnli': GLUETaskConfig(
        name='MNLI',
        dataset_name='glue/mnli',
        text_fields=('premise', 'hypothesis'),
        num_labels=3,
        metric='accuracy'
    ),
    'qqp': GLUETaskConfig(
        name='QQP',
        dataset_name='glue/qqp',
        text_fields=('question1', 'question2'),
        num_labels=2,
        metric='accuracy'
    ),
    'qnli': GLUETaskConfig(
        name='QNLI',
        dataset_name='glue/qnli',
        text_fields=('question', 'sentence'),
        num_labels=2,
        metric='accuracy'
    ),
    'sst2': GLUETaskConfig(
        name='SST-2',
        dataset_name='glue/sst2',
        text_fields=('sentence',),
        num_labels=2,
        metric='accuracy'
    ),
    'cola': GLUETaskConfig(
        name='CoLA',
        dataset_name='glue/cola',
        text_fields=('sentence',),
        num_labels=2,
        metric='matthews_correlation'
    ),
    'stsb': GLUETaskConfig(
        name='STS-B',
        dataset_name='glue/stsb',
        text_fields=('sentence1', 'sentence2'),
        num_labels=1,
        metric='pearson',
        is_regression=True
    ),
    'mrpc': GLUETaskConfig(
        name='MRPC',
        dataset_name='glue/mrpc',
        text_fields=('sentence1', 'sentence2'),
        num_labels=2,
        metric='accuracy'
    ),
    'rte': GLUETaskConfig(
        name='RTE',
        dataset_name='glue/rte',
        text_fields=('sentence1', 'sentence2'),
        num_labels=2,
        metric='accuracy'
    ),
}


class GLUEBenchmark:
    """
    Complete GLUE benchmark suite for Grammatical Transformers

    Usage:
        benchmark = GLUEBenchmark(
            model_name_or_path='bert-base-uncased',
            constituency_penalty=0.5,
            output_dir='./results'
        )
        results = benchmark.run_all_tasks()
    """

    def __init__(
        self,
        model_name_or_path: str = 'bert-base-uncased',
        constituency_penalty: float = 0.5,
        use_symmetry_loss: bool = True,
        symmetry_loss_weight: float = 0.1,
        output_dir: str = './glue_results',
        batch_size: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        seed: int = 42,
        max_seq_length: int = 128,
    ):
        """
        Args:
            model_name_or_path: Base model to load (e.g., 'bert-base-uncased')
            constituency_penalty: Cross-constituent attention penalty
            use_symmetry_loss: Whether to use symmetry loss
            symmetry_loss_weight: Weight for symmetry loss
            output_dir: Directory to save results
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            seed: Random seed
            max_seq_length: Maximum sequence length
        """
        if not DATASETS_AVAILABLE:
            raise ImportError(
                "datasets and transformers required for GLUE benchmarks. "
                "Install with: pip install datasets transformers"
            )

        self.model_name_or_path = model_name_or_path
        self.constituency_penalty = constituency_penalty
        self.use_symmetry_loss = use_symmetry_loss
        self.symmetry_loss_weight = symmetry_loss_weight
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.seed = seed
        self.max_seq_length = max_seq_length

        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def create_model(self, task_config: GLUETaskConfig) -> GrammaticalBertForSequenceClassification:
        """Create Grammatical BERT model for a specific task"""
        config = GrammaticalBertConfig.from_pretrained(
            self.model_name_or_path,
            num_labels=task_config.num_labels,
            constituency_penalty=self.constituency_penalty,
            use_symmetry_loss=self.use_symmetry_loss,
            symmetry_loss_weight=self.symmetry_loss_weight,
        )

        if task_config.is_regression:
            config.problem_type = "regression"

        model = GrammaticalBertForSequenceClassification(config)
        return model

    def preprocess_function(self, examples: Dict, task_config: GLUETaskConfig) -> Dict:
        """Tokenize examples for a specific task"""
        # Get text fields
        if len(task_config.text_fields) == 1:
            texts = examples[task_config.text_fields[0]]
            tokenized = self.tokenizer(
                texts,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
            )
        else:
            texts1 = examples[task_config.text_fields[0]]
            texts2 = examples[task_config.text_fields[1]]
            tokenized = self.tokenizer(
                texts1,
                texts2,
                padding='max_length',
                truncation=True,
                max_length=self.max_seq_length,
            )

        return tokenized

    def compute_metrics(self, eval_pred: EvalPrediction, task_config: GLUETaskConfig) -> Dict:
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred.predictions, eval_pred.label_ids

        if task_config.is_regression:
            # For STS-B (regression)
            from scipy.stats import pearsonr
            predictions = predictions.squeeze()
            pearson_corr = pearsonr(predictions, labels)[0]
            return {'pearson': pearson_corr}
        else:
            # Classification tasks
            predictions = np.argmax(predictions, axis=1)
            accuracy = (predictions == labels).mean()

            if task_config.metric == 'matthews_correlation':
                # For CoLA
                from sklearn.metrics import matthews_corrcoef
                mcc = matthews_corrcoef(labels, predictions)
                return {'matthews_correlation': mcc}
            else:
                return {'accuracy': accuracy}

    def run_task(self, task_name: str) -> Dict[str, float]:
        """
        Run benchmark on a single GLUE task

        Args:
            task_name: Name of GLUE task (e.g., 'mnli', 'sst2')

        Returns:
            Dictionary with evaluation metrics
        """
        if task_name not in GLUE_TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(GLUE_TASKS.keys())}")

        task_config = GLUE_TASKS[task_name]
        print(f"\n{'='*60}")
        print(f"Running GLUE Task: {task_config.name}")
        print(f"{'='*60}\n")

        # Load dataset
        print(f"Loading dataset: {task_config.dataset_name}")
        dataset = load_dataset(task_config.dataset_name.split('/')[0], task_config.dataset_name.split('/')[1])

        # Preprocess
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(
            lambda examples: self.preprocess_function(examples, task_config),
            batched=True,
        )

        # Create model
        print("Creating Grammatical BERT model...")
        model = self.create_model(task_config)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{self.output_dir}/{task_name}",
            eval_strategy="epoch",  # Changed from evaluation_strategy in newer transformers
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.num_epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=task_config.metric,
            seed=self.seed,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset['train'],
            eval_dataset=tokenized_dataset['validation'],
            compute_metrics=lambda eval_pred: self.compute_metrics(eval_pred, task_config),
        )

        # Train
        print("Training model...")
        trainer.train()

        # Evaluate
        print("Evaluating model...")
        results = trainer.evaluate()

        print(f"\n{task_config.name} Results:")
        print(f"  {task_config.metric}: {results[f'eval_{task_config.metric}']:.4f}")

        return results

    def run_all_tasks(self) -> Dict[str, Dict[str, float]]:
        """
        Run benchmark on all GLUE tasks

        Returns:
            Dictionary mapping task names to their results
        """
        all_results = {}

        print("\n" + "="*60)
        print("GLUE Benchmark - Grammatical Transformers")
        print("="*60)
        print(f"Model: {self.model_name_or_path}")
        print(f"Constituency Penalty: {self.constituency_penalty}")
        print(f"Symmetry Loss: {self.use_symmetry_loss}")
        print("="*60 + "\n")

        for task_name in GLUE_TASKS.keys():
            try:
                results = self.run_task(task_name)
                all_results[task_name] = results
            except Exception as e:
                print(f"\nError running task {task_name}: {e}")
                all_results[task_name] = {'error': str(e)}

        # Summary
        self._print_summary(all_results)

        return all_results

    def _print_summary(self, results: Dict[str, Dict[str, float]]):
        """Print summary of all results"""
        print("\n" + "="*60)
        print("GLUE Benchmark Summary")
        print("="*60 + "\n")

        # Calculate average score (excluding errors)
        valid_scores = []

        for task_name, task_results in results.items():
            if 'error' not in task_results:
                task_config = GLUE_TASKS[task_name]
                metric_key = f'eval_{task_config.metric}'
                if metric_key in task_results:
                    score = task_results[metric_key]
                    valid_scores.append(score)
                    print(f"{task_config.name:12} {task_config.metric:25} {score:.4f}")

        if valid_scores:
            avg_score = np.mean(valid_scores)
            print(f"\n{'Average Score':12} {' ':25} {avg_score:.4f}")

        print("\n" + "="*60 + "\n")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Run GLUE benchmark on Grammatical BERT')
    parser.add_argument('--model', default='bert-base-uncased', help='Base model')
    parser.add_argument('--task', default='all', help='Task to run (or "all")')
    parser.add_argument('--constituency_penalty', type=float, default=0.5, help='Constituency penalty')
    parser.add_argument('--output_dir', default='./glue_results', help='Output directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Create benchmark
    benchmark = GLUEBenchmark(
        model_name_or_path=args.model,
        constituency_penalty=args.constituency_penalty,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        seed=args.seed,
    )

    # Run
    if args.task == 'all':
        results = benchmark.run_all_tasks()
    else:
        results = benchmark.run_task(args.task)

    print("\nBenchmark complete!")


if __name__ == '__main__':
    main()

"""
Benchmarking suite for Grammatical Transformers

This module provides comprehensive benchmarks to evaluate:
1. GLUE tasks performance
2. Hallucination detection and reduction
3. Comparison with vanilla BERT
"""

from .glue_test import GLUEBenchmark
from .hallucination_test import HallucinationBenchmark
from .compare_vanilla import VanillaComparison

__all__ = [
    'GLUEBenchmark',
    'HallucinationBenchmark',
    'VanillaComparison',
]

"""
Setup file for Grammatical Transformers

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README from grammatical_transformers subdirectory
readme_file = Path(__file__).parent / "grammatical_transformers" / "README.md"
if readme_file.exists():
    with open(readme_file, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "Chomsky's Universal Grammar for Transformers"

setup(
    name="grammatical-transformers",
    version="0.1.0",
    author="Thiago Butignon",
    description="Chomsky's Universal Grammar for Transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nooa-ai/nooa-transformers",
    packages=find_packages(include=["grammatical_transformers", "grammatical_transformers.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "benchmarks": [
            "datasets>=2.10.0",
            "evaluate>=0.4.0",
            "scikit-learn>=1.2.0",
            "scipy>=1.10.0",
        ],
    },
)

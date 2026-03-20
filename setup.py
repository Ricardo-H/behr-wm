#!/usr/bin/env python3
"""
BehR-WM Package Setup
"""

from setuptools import setup, find_packages

setup(
    name="behr-wm",
    version="0.1.0",
    description="Beyond State Consistency: Behavior Consistency in Text-Based World Models",
    author="Anonymous",
    author_email="",
    url="https://anonymous.4open.science/r/behr-wm-787B",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "openai>=1.0.0",
        "requests",
        "tqdm",
        "pandas",
        "pyarrow",
    ],
    extras_require={
        "train": [
            "vllm>=0.6.0",
            "datasets",
        ],
        "eval": [
            "matplotlib",
            "aiohttp",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

#!/usr/bin/env python3
"""BehR-WM package setup."""

from setuptools import find_packages, setup

setup(
    name="behr-wm",
    version="0.2.0",
    description=(
        "BehR-WM: Behavior Consistency Reward for text-based world models."
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ricardo-H/behr-wm",
    license="Apache-2.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "openai>=1.0.0",
        "huggingface_hub>=0.24.0",
        "requests",
        "tqdm",
        "pandas",
        "pyarrow",
    ],
    extras_require={
        "train": [
            "vllm>=0.6.0",
            "datasets",
            "ray>=2.30.0",
            "wandb",
            "peft",
        ],
        "eval": [
            "matplotlib",
            "aiohttp",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

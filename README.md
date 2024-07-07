# ProtBench: Protein Language Modeling Benchmarking Library

Welcome to ProtBench! This library is designed to make benchmarking protein language models easy and modular. Whether you're adding new models, datasets, or using downstream models, this library has you covered. With support for embedding extraction and saving embeddings to disk, you can streamline your workflow and focus on what matters most: advancing your research.

## Features

- **Ease of Use**: Simple and intuitive API for benchmarking protein language models.
- **Modular Design**: Easily add new models and datasets for benchmarking.
- **Downstream Models**: Support for integrating and benchmarking downstream models (Currently, supports ConvBERT only).
- **LoRA Integration**: Use LoRA (Low-Rank Adaptation) for efficient benchmarking.
- **Embedding Extraction**: Extract embeddings and save them to disk for later use.

## Installation

To install the library, simply use pip:

```bash
git clone git@github.com:Proteinea/protbench.git
pip install -e .
```

## Quick Start

Here are some simple examples to get you started:
Example directories:

1. ESM2: protbench/examples/train_with_convbert_esm2.py
2. ANKH: protbench/examples/train_with_convbert.py
3. ESM2 with LoRA: protbench/examples/train_with_lora_esm2.py
4. Ankh with LoRA: protbench/examples/train_with_lora.py

## Documentation

Will be added soon.

## Contributing

We welcome contributions from the community. If you'd like to contribute, please fork the repository and submit a pull request.

# Encoder-Only Transformer from Scratch

This project demonstrates the implementation of an **Encoder-Only Transformer** from scratch using Python and PyTorch. The notebook provides a deep dive into building key components of a Transformer Encoder, leveraging only foundational libraries to enhance understanding of its architecture and working principles.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Components](#components)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

This repository focuses on constructing the Encoder part of a Transformer, which is widely used in Natural Language Processing (NLP) tasks such as text classification, sentiment analysis, and more. The implementation is educational and avoids the use of high-level frameworks like PyTorch or TensorFlow to emphasize the foundational concepts.

Key highlights:
- Pure Python and NumPy implementation.
- Modular structure for easy understanding and experimentation.
- Explanations and visualizations to aid conceptual learning.

---

## Features

- **Multi-Head Self-Attention**: Calculates attention scores across different heads for better information extraction.
- **Positional Encoding**: Adds positional information to the input embeddings.
- **Feed-Forward Network**: Applies a fully connected neural network to enhance feature representation.
- **Layer Normalization**: Normalizes inputs to improve training stability.
- **Residual Connections**: Facilitates gradient flow and prevents vanishing gradients.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mohammadreza-mohammadi94/Transformers-Hub.git
   cd Transformers-Hub/Encoder-Only\ From\ Scratch
   ```
---

## How It Works

The Transformer Encoder processes input sequences in the following steps:

1. **Input Embedding**: Maps tokenized input to dense vector representations.
2. **Positional Encoding**: Incorporates sequence order into embeddings.
3. **Multi-Head Self-Attention**: Captures contextual relationships between tokens.
4. **Feed-Forward Network**: Enhances token-wise feature extraction.
5. **Layer Stacking**: Repeats the above components for better representation learning.

The notebook provides in-depth explanations for each step, along with visual aids for better clarity.

---

## Components

The key components of the Encoder-Only Transformer implemented in the notebook are:

1. **Positional Encoding**: Adds positional information to input embeddings.
2. **Multi-Head Self-Attention**: Processes input to compute attention scores.
3. **Feed-Forward Neural Network**: Enhances feature extraction for each token.
4. **Layer Normalization and Residual Connections**: Ensures stability and gradient flow.
5. **Full Encoder Stack**: Combines all components into a cohesive module.

---

## Acknowledgments

- Inspired by the original Transformer paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
- Special thanks to the contributors of the Transformers community for fostering an environment of learning and exploration.

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use, modify, and distribute as per the terms of the license.

--- 

For additional details or queries, please feel free to reach out at [mr.mhmdi93@gmail.com](mailto:mr.mhmdi93@gmail.com).

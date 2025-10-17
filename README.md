# nanoBPE

A lightweight and efficient Byte Pair Encoding (BPE) implementation framework with CPU and GPU (CUDA) accelerated training support.

## 📌 Project Overview

nanoBPE is a clean and efficient implementation of the BPE (Byte Pair Encoding) algorithm, inspired by [Andrej Karpathy's minbpe](https://github.com/karpathy/minbpe).

**What is BPE?** Byte Pair Encoding is a data compression algorithm that builds a vocabulary by iteratively merging the most frequent byte pairs. In modern Large Language Models (such as the GPT series), BPE is the standard tokenization algorithm.

## ⚡ Core Features

- **Efficient Implementation**: Clean and understandable code design, suitable for learning and research
- **Dual-End Support**: Both CPU-based pure Python and PyTorch GPU-accelerated implementations
- **Significant Acceleration**: GPU version is **20x faster** than CPU
- **Flexible Configuration**: Support for custom vocabulary size and training parameters
- **Complete Toolchain**: Includes model saving, loading, and inference functionality

## 📊 Performance Comparison

Performance on the Taylor Swift dataset (approximately 10MB):

| Environment | Time | GPU | 
|-------------|------|------|
| CPU | 13.3s | ❌ |
| GPU (RTX 3080) | 0.7s | ✅ |
| **Speedup** | **20x** | - |

> 💡 The acceleration effect becomes even more significant（ > 100x ） on larger datasets (>100MB)

## 🚀 Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### CPU Training (Pure Python)

```bash
python train.py
```

### GPU Training (PyTorch + CUDA)

```bash
python train_torch.py
```

## 📋 Dependencies

- `torch` - Deep learning framework for GPU acceleration
- `regex` - Advanced regular expression support
- `tiktoken` - OpenAI's tokenizer library reference
- `pytest` - Unit testing framework

## 🏗️ Project Structure

```
nanobpe/
├── train.py                 # CPU version training script
├── train_torch.py          # GPU version training script
├── nanobpe/                # Core implementation module
│   ├── basic_torch.py      # PyTorch GPU accelerated implementation
│   └── ...
├── tests/                  # Test cases and example data
├── models/                 # Trained model storage directory
└── requirements.txt        # Dependency declaration
```

## 💡 Usage Example

```python
from nanobpe import BasicTorchTokenizer

# Create tokenizer
tokenizer = BasicTorchTokenizer()

# Train model with vocabulary size of 512
text = open("data.txt", "r", encoding="utf-8").read()
tokenizer.train(text, vocab_size=512, verbose=True)

# Save model
tokenizer.save("models/my_tokenizer")

# Use model for encoding
tokens = tokenizer.encode("Hello, World!")
```

## 📚 Learning Resources

- [BPE Algorithm Deep Dive](https://github.com/karpathy/minbpe)
- [Tokenization Algorithm Survey](https://huggingface.co/docs/transformers/tokenizer_summary)
- [CUDA Optimization Guide](https://docs.nvidia.com/cuda/)

## 📝 License

This project is licensed under the MIT License.

---

**Note**: This project is primarily intended for educational and research purposes, to help understand the principles of BPE algorithms and GPU acceleration.
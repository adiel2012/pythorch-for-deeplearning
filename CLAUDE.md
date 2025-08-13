# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a comprehensive PyTorch tutorial written in LaTeX format. The project consists of:

- `pytorch_tutorial.tex` - Main LaTeX document containing a complete PyTorch learning guide covering fundamentals to advanced neural networks
- `test_tikz.tex` - Test document for verifying TikZ diagram functionality
- `_minted-pytorch_tutorial/` - Directory containing minted package cache files for syntax highlighting

## Document Structure

The main tutorial is organized into the following chapters:

1. **Mathematical Foundations** - Linear algebra, probability distributions, information theory, optimization theory
2. **Fundamental Tensor Operations** - Creating tensors, properties and manipulation
3. **Mathematical Operations** - Basic arithmetic, activation functions
4. **Automatic Differentiation** - Gradient computation with `requires_grad`
5. **Convolutional Neural Networks** - Conv layers, pooling, activations
6. **Neural Network Building Blocks** - Linear layers, embeddings, normalization
7. **Essential Deep Learning Utilities** - Gradient clipping, model utilities, parameter initialization
8. **Recurrent Neural Networks** - LSTM/GRU, attention mechanisms, transformers
9. **Advanced Operations** - Functional operations, advanced tensor operations
10. **Optimization and Training** - Optimizers, loss functions
11. **Sampling and Generation** - Random sampling techniques
12. **Generative Models** - GANs, VAEs, advanced variants
13. **Complete Examples** - Neural networks, language models, transformers
14. **PyTorch 2.x Features** - torch.compile, mixed precision, optimizations
15. **Best Practices** - Memory management, device handling, debugging

## Build Commands

To compile the LaTeX documents:

```bash
# Main tutorial (requires shell-escape for minted package)
pdflatex -shell-escape pytorch_tutorial.tex

# Simple test document
pdflatex test_tikz.tex

# Alternative LaTeX engines are available:
xelatex -shell-escape pytorch_tutorial.tex
lualatex -shell-escape pytorch_tutorial.tex
```

## LaTeX Configuration

The document uses several key packages:
- `minted` for Python syntax highlighting (requires `-shell-escape` flag)
- `tikz` and `pgfplots` for tensor visualization diagrams
- `hyperref` for navigation and cross-references
- Custom environments for Python code blocks with output formatting

## Visual Enhancements

The document includes comprehensive TikZ visualizations:

### Neural Network Visualizations
- **Convolution Operations**: Visual diagrams showing input feature maps, kernels, and output feature maps with stride and padding illustrations
- **CNN Architecture**: Complete convolutional neural network pipeline from input to classification
- **Pooling Operations**: Max pooling visualization with detailed 4×4 to 2×2 transformation examples
- **Multilayer Perceptron**: Fully connected neural network architecture showing layers, connections, and activation functions

### Advanced Architecture Diagrams
- **LSTM Networks**: Time-series visualization showing hidden states, cell states, and recurrent connections across time steps
- **Multi-Head Attention**: Transformer attention mechanism with Query, Key, Value projections and multi-head processing
- **Computational Graphs**: Automatic differentiation visualization showing forward and backward passes

### Mathematical Concepts
- **Tensor Shape Transformations**: 1D, 2D, and 3D tensor visualizations with shape annotations
- **Gradient Flow**: Computational graph diagrams for understanding backpropagation

### TikZ Style Definitions
The document includes predefined styles for consistent visualization:
- `neuron`, `input`, `hidden`, `output`: Neural network node styles
- `conv`, `pool`, `fc`: CNN layer styles  
- `attention`, `head`, `qkv`: Transformer component styles
- `tensor1d`, `tensor2d`, `tensor3d`, `tensor4d`: Tensor visualization styles

## Working with the Tutorial

- The document includes extensive Python code examples using the `minted` package
- TikZ diagrams are used to visualize tensor shapes and neural network architectures
- Code blocks use a custom `pythoncode` environment with syntax highlighting
- Output comments are formatted using the `\pyoutput{}` command
- All visualizations are designed to complement the code examples and enhance understanding
- The document now contains 105 pages with comprehensive visual aids for learning PyTorch concepts

## Google Colab Notebooks

The `notebooks/` directory contains interactive Jupyter notebooks designed for Google Colab:

### Available Notebooks

1. **01_fundamental_tensor_operations.ipynb**
   - Tensor creation, manipulation, and basic operations
   - Interactive exercises with broadcasting and indexing
   - Comprehensive visualizations of tensor operations

2. **02_mathematical_operations.ipynb** 
   - Advanced mathematical functions and linear algebra
   - Activation function comparisons and visualizations
   - Numerical stability demonstrations

3. **03_automatic_differentiation.ipynb**
   - Autograd system and computational graphs
   - Gradient computation and backpropagation
   - Manual gradient descent implementation

4. **04_convolutional_neural_networks.ipynb**
   - Complete CNN architecture building and training
   - Feature visualization and filter analysis
   - Image classification pipeline

5. **05_complete_examples.ipynb**
   - End-to-end deep learning projects
   - Image classification, regression, autoencoders, and GANs
   - Production-ready training loops and evaluation

### Notebook Features

- **Interactive Learning**: Hands-on coding with immediate feedback
- **Google Colab Ready**: One-click execution in browser
- **Comprehensive Examples**: From basics to complete projects
- **Visualizations**: Matplotlib plots and interactive demonstrations
- **Best Practices**: Production-ready code patterns
- **Progressive Difficulty**: Structured learning path

### Getting Started with Notebooks

1. **Google Colab (Recommended)**:
   - Click "Open in Colab" badges in notebook files
   - Automatic PyTorch installation and GPU access
   - No local setup required

2. **Local Jupyter**:
   - Install requirements: `torch`, `torchvision`, `matplotlib`, `seaborn`
   - Launch with `jupyter notebook`
   - Run notebooks in order for best learning experience

### Learning Path

**Recommended sequence**:
1. Fundamental Tensor Operations (2-3 hours)
2. Mathematical Operations (1-2 hours) 
3. Automatic Differentiation (2-3 hours)
4. Convolutional Neural Networks (3-4 hours)
5. Complete Examples (4-6 hours)

**Total estimated time**: 12-18 hours for complete mastery

## Git Configuration

The repository includes a comprehensive `.gitignore` file that handles:

### LaTeX Files
- Auxiliary files: `*.aux`, `*.log`, `*.out`, `*.toc`, `*.synctex.gz`
- Minted cache: `_minted-*/`, `*.pygtex`, `*.pygstyle`
- BibTeX files: `*.bbl`, `*.blg`, `*.bcf`
- Intermediate files: `*.dvi`, `*.ps`

### Python & ML Files
- Python cache: `__pycache__/`, `*.py[cod]`, `*.so`
- PyTorch models: `*.pth`, `*.pt`, `*.ckpt`
- Training artifacts: `checkpoints/`, `runs/`, `logs/`, `wandb/`
- Datasets: `data/`, `datasets/`, `*.csv`, `*.json`, `*.pkl`
- Virtual environments: `.venv/`, `venv/`, `.conda/`

### Development Environment
- IDE files: `.vscode/`, `.idea/`, `*.sublime-*`
- Editor temporaries: `*.swp`, `*.swo`, `*~`
- OS files: `.DS_Store`, `Thumbs.db`, `Desktop.ini`
- Cache and temporary files: `*.tmp`, `*.cache`, `.pytest_cache/`

### Notes
- PDF files are currently tracked (change by uncommenting `# *.pdf` in .gitignore)
- Main source files (`.tex`) and documentation (`.md`) are always tracked
- The `.gitignore` is organized in sections for easy maintenance
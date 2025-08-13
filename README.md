# PyTorch for Deep Learning

A comprehensive educational resource for learning PyTorch and deep learning fundamentals, combining theoretical foundations with practical implementation.

[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue)](https://github.com/adiel2012/pythorch-for-deeplearning.git)
[![License](https://img.shields.io/badge/License-Educational-green)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-red)](https://pytorch.org/)

## 📚 Project Overview

This project provides a complete learning pathway for PyTorch and deep learning, featuring:
- **Comprehensive PDF Tutorial**: 105-page detailed guide with mathematical foundations
- **Interactive Colab Notebooks**: Hands-on exercises and practical implementations
- **Visual Learning**: TikZ diagrams and matplotlib visualizations
- **Progressive Learning**: From tensor basics to advanced architectures

## 📁 Repository Structure

### 📖 Main Tutorial
- **[`pytorch_tutorial.pdf`](pytorch_tutorial.pdf)** - Complete 105-page tutorial covering:
  - PyTorch fundamentals and tensor operations
  - Mathematical foundations and automatic differentiation
  - Neural network architectures (CNNs, RNNs, Transformers)
  - Advanced topics and optimization techniques
  - Real-world applications and best practices

- **[`pytorch_tutorial.tex`](pytorch_tutorial.tex)** - LaTeX source with enhanced TikZ visualizations:
  - Convolution operation diagrams
  - CNN architecture visualizations
  - LSTM and attention mechanism illustrations
  - Mathematical notation and computational graphs

### 💻 Interactive Notebooks

#### **[`notebooks/`](notebooks/)** - Google Colab Ready Notebooks

| Notebook | Description | Topics Covered | Open in Colab |
|----------|-------------|----------------|---------------|
| [**01_fundamental_tensor_operations.ipynb**](notebooks/01_fundamental_tensor_operations.ipynb) | Master PyTorch tensor basics | Tensor creation, manipulation, broadcasting, aggregation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/01_fundamental_tensor_operations.ipynb) |
| [**02_mathematical_operations.ipynb**](notebooks/02_mathematical_operations.ipynb) | Advanced math and activation functions | Trigonometry, linear algebra, activation comparisons | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/02_mathematical_operations.ipynb) |
| [**03_automatic_differentiation.ipynb**](notebooks/03_automatic_differentiation.ipynb) | Understanding PyTorch's autograd | Computational graphs, backpropagation, gradient management | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/03_automatic_differentiation.ipynb) |
| [**04_convolutional_neural_networks.ipynb**](notebooks/04_convolutional_neural_networks.ipynb) | Complete CNN implementation | Conv layers, pooling, architecture design, CIFAR-10 training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/04_convolutional_neural_networks.ipynb) |
| [**05_complete_examples.ipynb**](notebooks/05_complete_examples.ipynb) | End-to-end projects | Classification, regression, autoencoders, GANs | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/05_complete_examples.ipynb) |

**🎓 Learning Progression:**
1. [![Start Here](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/01_fundamental_tensor_operations.ipynb) **Begin with Tensor Operations**
2. See [`notebooks/README.md`](notebooks/README.md) for detailed learning guide and prerequisites

### 🛠️ Project Configuration

- **[`CLAUDE.md`](CLAUDE.md)** - Complete project documentation for Claude Code:
  - Build instructions and project overview
  - Development workflow and architecture details
  - Visual enhancements and notebook integration
  - Git repository management guidelines

- **[`.gitignore`](.gitignore)** - Comprehensive ignore patterns:
  - LaTeX compilation artifacts and auxiliary files
  - Python/ML temporary files and model checkpoints
  - Development environment and IDE configurations
  - System-specific files and cache directories

## 🚀 Getting Started

### Option 1: Interactive Learning (Recommended)
1. **Browse Notebooks**: Visit the [`notebooks/`](notebooks/) directory
2. **Open in Colab**: Click any "Open in Colab" badge below
3. **Start Learning**: Begin with fundamental tensor operations
4. **Progress Sequentially**: Follow the numbered learning path

**🚀 Quick Launch Notebooks:**
- [![Notebook 1](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/01_fundamental_tensor_operations.ipynb) **Tensor Fundamentals** - Start here!
- [![Notebook 2](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/02_mathematical_operations.ipynb) **Mathematical Operations** - Advanced functions
- [![Notebook 3](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/03_automatic_differentiation.ipynb) **Autograd System** - Gradients & backprop
- [![Notebook 4](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/04_convolutional_neural_networks.ipynb) **CNNs** - Computer vision
- [![Notebook 5](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/05_complete_examples.ipynb) **Complete Projects** - End-to-end implementations

### Option 2: Complete PDF Study
1. **Download**: [`pytorch_tutorial.pdf`](pytorch_tutorial.pdf)
2. **Read Systematically**: 17 chapters from basics to advanced topics
3. **Reference Visuals**: TikZ diagrams for complex concepts
4. **Practice**: Use notebooks for hands-on experience

### Option 3: Local Development
```bash
# Clone repository
git clone https://github.com/adiel2012/pythorch-for-deeplearning.git
cd pythorch-for-deeplearning

# Install dependencies
pip install torch torchvision torchaudio matplotlib numpy jupyter

# Launch notebooks
jupyter notebook notebooks/

# Compile LaTeX (optional)
pdflatex -shell-escape pytorch_tutorial.tex
```

## 📋 Learning Path

### **Beginner Track** (6-10 hours)
1. 📖 Read Chapters 1-5 in PDF
2. 💻 Complete notebooks 01-03
3. 🎯 Practice tensor operations and autograd

### **Intermediate Track** (10-15 hours)
1. 📖 Read Chapters 6-12 in PDF
2. 💻 Complete notebook 04 (CNNs)
3. 🎯 Build and train your first neural network

### **Advanced Track** (15-25 hours)
1. 📖 Read Chapters 13-17 in PDF
2. 💻 Complete notebook 05 (Complete projects)
3. 🎯 Implement advanced architectures and projects

## 🎨 Visual Learning Features

### TikZ Visualizations
- **Convolution Operations**: Step-by-step kernel applications
- **CNN Architectures**: Layer-by-layer network diagrams
- **Pooling Operations**: Max/average pooling illustrations
- **Neural Networks**: Fully connected layer visualizations
- **LSTM Components**: Memory cells and gate mechanisms
- **Attention Mechanisms**: Multi-head attention diagrams

### Interactive Plots
- **Activation Functions**: Behavior and gradient analysis
- **Training Curves**: Loss and accuracy monitoring
- **Feature Maps**: CNN layer output visualizations
- **Gradient Flow**: Backpropagation analysis

## 🧠 Topics Covered

### **Core PyTorch** (Chapters 1-6)
- Tensor operations and broadcasting
- Automatic differentiation (autograd)
- Neural network modules (`nn.Module`)
- Optimization algorithms
- Data loading and preprocessing

### **Deep Learning Architectures** (Chapters 7-13)
- Feedforward neural networks
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs/LSTMs)
- Transformer architectures
- Generative models (VAEs, GANs)

### **Advanced Topics** (Chapters 14-17)
- Transfer learning and fine-tuning
- Model optimization and quantization
- Deployment strategies
- Best practices and debugging

## 🔧 Prerequisites

**Required:**
- Basic Python programming
- High school mathematics (algebra, calculus basics)
- Familiarity with NumPy arrays

**Recommended:**
- Linear algebra fundamentals
- Basic machine learning concepts
- Experience with Jupyter notebooks

## 🌟 Key Features

- ✅ **Complete Learning Path**: From basics to advanced implementations
- ✅ **Theory + Practice**: PDF theory with interactive coding
- ✅ **Visual Learning**: Comprehensive diagrams and visualizations
- ✅ **Modern PyTorch**: Up-to-date syntax and best practices
- ✅ **Production Ready**: Real-world examples and deployable code
- ✅ **Self-Paced**: Learn at your own speed with clear progression

## 🤝 Contributing

We welcome contributions! Here's how to get involved:

1. **Report Issues**: Found a bug or unclear explanation? Open an issue
2. **Suggest Improvements**: Ideas for better explanations or examples
3. **Add Examples**: Contribute additional practice problems or projects
4. **Fix Typos**: Help improve documentation quality

## 📄 License

This educational resource is provided for learning purposes. See individual files for specific licensing terms.

## 🔗 Additional Resources

**Official PyTorch:**
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch Forums](https://discuss.pytorch.org/)

**Community:**
- [PyTorch GitHub](https://github.com/pytorch/pytorch)
- [Papers With Code](https://paperswithcode.com/)
- [Distill.pub](https://distill.pub/)

## 📞 Support

- 🐛 **Issues**: Use GitHub issues for bugs and questions
- 💡 **Discussions**: Share ideas and ask questions in discussions
- 📧 **Contact**: Reach out for collaboration opportunities

---

**Start your PyTorch journey today! 🚀**

*Master deep learning one tensor at a time.*

<!-- Repository Stats -->
![Repository Size](https://img.shields.io/github/repo-size/adiel2012/pythorch-for-deeplearning)
![Last Commit](https://img.shields.io/github/last-commit/adiel2012/pythorch-for-deeplearning)
![Languages](https://img.shields.io/github/languages/count/adiel2012/pythorch-for-deeplearning)
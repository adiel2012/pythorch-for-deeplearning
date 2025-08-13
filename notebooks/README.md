# PyTorch for Deep Learning - Colab Notebooks

This directory contains Google Colab notebooks that complement the PyTorch tutorial PDF. Each notebook provides hands-on, interactive learning experiences for different aspects of PyTorch and deep learning.

## 📚 Notebook Overview

### [01_fundamental_tensor_operations.ipynb](01_fundamental_tensor_operations.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/01_fundamental_tensor_operations.ipynb)

**Learning Objectives:**
- Master tensor creation and manipulation
- Understand tensor properties (shape, dtype, device)
- Learn indexing, slicing, and broadcasting
- Practice basic mathematical operations

**Key Topics:**
- Tensor creation methods (`torch.zeros`, `torch.randn`, etc.)
- Reshaping and dimension manipulation
- Broadcasting rules and examples
- Aggregation operations (sum, mean, max)

---

### [02_mathematical_operations.ipynb](02_mathematical_operations.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/02_mathematical_operations.ipynb)

**Learning Objectives:**
- Explore advanced mathematical functions
- Understand activation functions and their properties
- Learn about numerical stability
- Visualize function behaviors and gradients

**Key Topics:**
- Trigonometric, exponential, and logarithmic functions
- Linear algebra operations (matrix multiplication, eigenvalues)
- Activation functions (ReLU, Sigmoid, Tanh, GELU, etc.)
- Softmax and log-softmax for classification
- Numerical stability techniques

---

### [03_automatic_differentiation.ipynb](03_automatic_differentiation.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/03_automatic_differentiation.ipynb)

**Learning Objectives:**
- Master PyTorch's autograd system
- Understand computational graphs
- Learn gradient management techniques
- Implement manual gradient descent

**Key Topics:**
- `requires_grad` and gradient tracking
- Computational graph construction and traversal
- Backpropagation mechanics
- Gradient accumulation and zeroing
- Higher-order derivatives
- Context managers (`torch.no_grad()`, `torch.inference_mode()`)

---

### [04_convolutional_neural_networks.ipynb](04_convolutional_neural_networks.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/04_convolutional_neural_networks.ipynb)

**Learning Objectives:**
- Build and train convolutional neural networks
- Understand convolution and pooling operations
- Visualize learned features and filters
- Apply CNNs to image classification

**Key Topics:**
- Conv2d layers and parameter calculations
- Pooling operations (Max, Average, Adaptive)
- CNN architecture design principles
- Feature visualization and interpretation
- Complete training pipeline
- Data augmentation and regularization

---

### [05_complete_examples.ipynb](05_complete_examples.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adiel2012/pythorch-for-deeplearning/blob/main/notebooks/05_complete_examples.ipynb)

**Learning Objectives:**
- Implement end-to-end deep learning projects
- Learn best practices for model development
- Understand different problem types and approaches
- Practice complete ML pipelines

**Key Projects:**
1. **Image Classification**: Complete CNN pipeline with validation
2. **Regression**: Neural network for continuous prediction
3. **Autoencoder**: Dimensionality reduction and reconstruction
4. **Simple GAN**: Generative adversarial network for data generation

---

## 🚀 Getting Started

### Option 1: Google Colab (Recommended)
1. Click the "Open in Colab" badge for any notebook
2. Sign in to your Google account
3. Run the setup cell to install PyTorch
4. Follow along with the examples

### Option 2: Local Setup
1. Clone the repository
2. Install requirements: `pip install torch torchvision matplotlib seaborn scikit-learn`
3. Launch Jupyter: `jupyter notebook`
4. Open the desired notebook

## 📋 Prerequisites

**Required Knowledge:**
- Basic Python programming
- Familiarity with NumPy
- Basic understanding of machine learning concepts

**Python Libraries:**
- PyTorch (`torch`, `torchvision`)
- NumPy and Matplotlib
- Scikit-learn (for utilities)
- Seaborn (for visualization)

## 🎯 Learning Path

**Recommended Order:**
1. Start with **Fundamental Tensor Operations** for PyTorch basics
2. Progress to **Mathematical Operations** for function understanding
3. Master **Automatic Differentiation** for gradient concepts
4. Build **Convolutional Neural Networks** for computer vision
5. Complete **End-to-End Examples** for practical experience

**Time Estimate:**
- Each notebook: 1-2 hours
- Complete series: 6-10 hours
- Additional practice: 5-15 hours

## 🛠️ Features

**Interactive Learning:**
- ✅ Hands-on code examples
- ✅ Visualizations and plots
- ✅ Practice exercises
- ✅ Real-time feedback
- ✅ Progressive difficulty

**Best Practices:**
- ✅ Proper error handling
- ✅ GPU/CPU compatibility
- ✅ Memory management
- ✅ Reproducible results
- ✅ Clean code structure

## 💡 Tips for Success

1. **Run Everything**: Execute every code cell to understand the flow
2. **Experiment**: Modify parameters and observe changes
3. **Visualize**: Pay attention to plots and their interpretations
4. **Practice**: Complete all exercises before moving on
5. **Debug**: Use print statements to understand tensor shapes
6. **Document**: Take notes on key concepts and insights

## 🔍 Common Issues & Solutions

**GPU Access in Colab:**
- Navigate to Runtime → Change runtime type → Hardware accelerator → GPU

**Memory Issues:**
- Use smaller batch sizes
- Clear intermediate variables with `del variable_name`
- Restart runtime if needed

**Import Errors:**
- Run the installation cells at the beginning of each notebook
- Check that all required packages are installed

## 📚 Additional Resources

**Official Documentation:**
- [PyTorch Docs](https://pytorch.org/docs/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

**Community:**
- [PyTorch Forums](https://discuss.pytorch.org/)
- [PyTorch GitHub](https://github.com/pytorch/pytorch)

**Books & Courses:**
- "Deep Learning with PyTorch" by Eli Stevens
- "Programming PyTorch for Deep Learning" by Ian Pointer
- Fast.ai Practical Deep Learning Course

## 🤝 Contributing

Found an issue or want to improve a notebook?
1. Report bugs via GitHub issues
2. Suggest improvements
3. Submit pull requests
4. Share your learning experience

## 📄 License

These notebooks are provided for educational purposes and complement the main PyTorch tutorial document.

---

**Happy Learning! 🎉**

*Master PyTorch one tensor at a time!*
# MNIST Neural Network Walkthrough

## Overview
This project breaks down neural networks into simple steps by walking through the process of building one from scratch using Python and NumPy. The notebook trains a neural network on the MNIST Dataset, a collection of 70,000 handwritten digit images (0-9), each 28x28 pixels in size. By following this guide, you'll gain a solid understanding of how neural networks work and create a model capable of classifying digits with over 90% accuracy.

## Features
- **Introduction to Neural Networks:** Learn the fundamental principles behind neural networks and their layered structure.
- **Custom Implementation:** Build a feedforward neural network from scratch using Python and NumPy.
- **Training and Testing:** Train the model on the MNIST dataset, fine-tune its parameters, and evaluate its performance.
- **Visualization:** Gain insights into the network's behavior and classification accuracy with intuitive visualizations.

## Neural Network Structure
The feedforward neural network implemented in this notebook has the following architecture:
- **Input Layer:** 784 neurons (28x28 pixel values per MNIST image).
- **Hidden Layer 1:** 32 neurons with ReLU activation.
- **Hidden Layer 2:** 16 neurons with ReLU activation.
- **Output Layer:** 10 neurons (one for each digit, 0-9) with softmax activation.

## Dependencies
To run this notebook, ensure you have the following installed:
- Python 3.7 or higher
- NumPy
- Matplotlib
- Jupyter Notebook or Jupyter Lab

You can install the dependencies using:
```
pip install numpy matplotlib notebook
```

## How to Use

1. Clone the repository
   ```
   git clone https://github.com/dinosaur-oatmeal/MNIST-Practice.git
   cd MNIST-Walkthrough
   ```
2. Install the required dependencies
3. Open the notebook:
   ```
   jupyter notebook MNIST-Walkthrough.ipynb
   ```
4. Run the cells step-by-step to follow the implementation of the neural network

## Results
By the end of the notebook, you will:
* Understand the fundamentals of training neural networks from scratch.
* Train a model to classify handwritten digits with over 90% accuracy.
* Visualize the performance metrics and insights of the trained network.

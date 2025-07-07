# Dynamic Neural Networks: Implementations from Scratch and with Frameworks

## Overview

This repository showcases the implementation and analysis of fundamental and advanced neural network concepts in Python. It contains three distinct Jupyter Notebooks, each focusing on a different aspect of dynamic neural networks:

1.  **Scaled Dot-Product Attention**: A from-scratch implementation of the core mechanism behind the Transformer architecture, demonstrating its function with synthetic data.
2.  **Network Pruning**: A practical application of model optimization, where a pre-trained neural network is pruned to reduce complexity while maintaining performance on the Fashion-MNIST dataset.
3.  **Feedforward Neural Network from Scratch**: A foundational implementation of a multi-layer perceptron (MLP) using only NumPy, trained on the Iris dataset and benchmarked against a Keras equivalent.

## Project Structure

```
.
└── Dynamic Neural Networks.ipynb   # Jupyter Notebook containing all three implementations.
```
---

## Part 1: Scaled Dot-Product Attention

### 1.1. Objective

To implement the scaled dot-product attention mechanism as a standalone Python function using only NumPy. This exercise demonstrates a fundamental understanding of the building block that powers modern Transformer models.

### 1.2. Methodology

The attention mechanism is defined by the formula:
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
- **Implementation**: The function was built to compute the dot product of Query (Q) and Key (K) matrices, scale the result by the square root of the key dimension ($d_k$), apply a softmax function to obtain attention weights, and finally compute a weighted sum of the Value (V) matrix.
- **Validation**: The function was tested on randomly generated synthetic data to visualize its behavior in an unstructured context.

### 1.3. Key Findings

- **Correct Functionality**: The implementation correctly produced attention weights and a corresponding output matrix.
- **Uniform Attention on Synthetic Data**: With random input, the attention weights were nearly uniform across all key positions for each query. This is the expected behavior, as there are no inherent relationships for the model to "attend" to.
- **Visualization**: A heatmap of the attention weights effectively illustrated this uniform distribution, providing a useful tool for interpreting the mechanism's focus.

---

## Part 2: Network Pruning on Fashion-MNIST

### 2.1. Objective

To optimize a pre-trained neural network by applying a network pruning technique. This demonstrates a practical method for reducing model complexity, which is crucial for deployment on resource-constrained devices.

### 2.2. Methodology

1.  **Baseline Model**: A simple feedforward neural network was built and trained on the Fashion-MNIST dataset for 10 epochs.
2.  **Pruning Technique**: **Magnitude-based threshold pruning** was applied. All weights in the dense layers (excluding the final output layer) with an absolute magnitude below a threshold of **0.1** were set to zero.
3.  **Retraining (Fine-Tuning)**: The pruned model was retrained for an additional 5 epochs to allow the network to recover from the information loss caused by pruning and adapt to its new, sparser architecture.
4.  **Evaluation**: The model's accuracy and sparsity were measured at three stages: original, pruned (before retraining), and pruned-and-retrained.

### 2.3. Key Findings

- **Original Accuracy**: 88.09%
- **Pruned Accuracy (before retraining)**: 88.10% (negligible change)
- **Final Accuracy (after retraining)**: **88.29%**
- **Sparsity**: Pruning introduced **6.7%** sparsity into the model by removing 7,315 weights, with a minimal impact on performance.

The results show that a moderate level of pruning can effectively reduce model parameters without sacrificing—and in this case, even slightly improving—accuracy after fine-tuning.

---

## Part 3: Feedforward Neural Network from Scratch

### 3.1. Objective

To build, train, and evaluate a simple feedforward neural network for multi-class classification using only Python and NumPy. This exercise solidifies foundational knowledge of neural network mechanics, including forward propagation, backpropagation, and gradient descent.

### 3.2. Methodology

1.  **From-Scratch Implementation**:
    - A `NeuralNetwork` class was created with methods for forward and backward passes.
    - **Activation Function**: Sigmoid.
    - **Loss Function**: Mean Squared Error (MSE).
    - **Optimization**: Standard (batch) gradient descent.
2.  **Dataset**: The classic Iris dataset was used for this multi-class classification task.
3.  **Benchmarking**: The from-scratch model's performance was compared against a simple Keras model with a similar architecture trained on the same data.
4.  **Evaluation**: Both models were evaluated using a classification report, confusion matrix, ROC curves, and an overall metrics comparison bar chart.

### 3.3. Key Findings

- **Scratch Network Accuracy**: **97%**
- **Keras Model Accuracy**: **93%**

The from-scratch network surprisingly outperformed the Keras model on this simple dataset. This is likely due to the dataset's small size and linear separability, where the Keras model's dropout layers may have introduced unnecessary regularization, leading to slight underfitting. The from-scratch implementation demonstrated a solid grasp of the underlying principles and achieved excellent performance.

## How to Run the Analysis

1.  **Prerequisites**: Ensure you have a Python environment with the following libraries installed:
    ```
    pip install numpy pandas matplotlib scikit-learn tensorflow altair seaborn plotly
    ```
2.  **Execution**: Open `Dynamic Neural Networks.ipynb` in a Jupyter environment (like Jupyter Lab or Google Colab) and run the cells sequentially. Each of the three parts is self-contained.

## Critical Analysis and Future Work

This repository provides a strong foundation, but there are clear avenues for expansion and improvement:

- **Attention Mechanism**: The next logical step would be to integrate the standalone attention function into a full Transformer encoder or decoder block and train it on a real NLP task to observe meaningful attention patterns.
- **Pruning**: The magnitude-based pruning used is a simple approach. More advanced techniques like **structured pruning** (removing entire neurons or channels) or **iterative pruning** could be explored to achieve higher sparsity with less performance degradation.
- **Scratch Network**: The from-scratch MLP could be enhanced by implementing more advanced features, such as different activation functions (ReLU, Leaky ReLU), more sophisticated optimizers (Adam, RMSprop), and batch/mini-batch gradient descent to handle larger datasets.

## License

This project is open-source and available under the MIT License.

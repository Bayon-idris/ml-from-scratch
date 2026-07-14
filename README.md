# ML From Scratch

A collection of machine learning and deep learning models implemented from scratch using NumPy, with a focus on understanding the mathematics, algorithms, and training mechanisms behind modern AI systems.

## Project Purpose

The goal of this repository is not to achieve state-of-the-art performance or build production-ready models.

Instead, the objective is to understand how machine learning and deep learning algorithms work internally — first by implementing them from scratch using NumPy, then by comparing that understanding against high-level frameworks.

This is a personal learning project, not a research project.

Each notebook focuses on core concepts including:

- Forward propagation
- Backpropagation
- Gradient descent
- Loss functions
- Parameter optimization
- Neural network and CNN architectures

## Project Progress

| Notebook | Approach | Status |
|-----------|-----------|--------|
| 01 - Single-Layer Perceptron | NumPy from scratch | ✅ Completed |
| 02 - Logistic Regression | NumPy from scratch | ✅ Completed |
| 03 - Neural Network, One Hidden Layer | NumPy from scratch | ✅ Completed |
| 04 - Deep Neural Network (L Layers) | NumPy from scratch | ✅ Completed |
| 05 - Convolution Operations | NumPy from scratch | ✅ Completed |
| 06 - Convolutional Neural Network (MNIST) | TensorFlow/Keras | ✅ Completed |
| 07 - Deep CNN (Cats vs Dogs) | TensorFlow/Keras | ✅ Completed |
| 08 - Recurrent Neural Networks | TensorFlow/Keras | 🚧 Planned |

## Key Result: Why CNNs?

Notebooks 03 and 04 attempt cat vs. dog classification using fully-connected networks implemented from scratch, reaching only ~56-58% test accuracy — barely above chance. Notebook 07 tackles the same type of problem with a CNN and reaches ~80% test accuracy. This gap is a direct, hands-on illustration of why convolutional architectures are necessary for image tasks.

## Learning Path

1. Single-Layer Perceptron
2. Logistic Regression
3. Neural Network with One Hidden Layer
4. Deep Neural Network (L Layers)
5. Convolutional Operations (from scratch)
6. Convolutional Neural Networks (Keras)
7. Deep CNN – Cats vs Dogs (Keras)
8. Recurrent Neural Networks (planned, exploratory — not a primary focus)

Notebooks 01-05 implement every operation manually in NumPy (forward pass, backward pass, gradient updates) to understand the underlying math. From notebook 06 onward, the focus shifts to building practical, trainable CNN architectures using TensorFlow/Keras, since the goal there is applying the concepts rather than re-deriving them.

## Technologies

- Python, NumPy, Matplotlib, Scikit-learn (notebooks 01-05)
- TensorFlow / Keras (notebooks 06-07)

## Philosophy

This project is about understanding how AI models work under the hood before relying on high-level frameworks. My primary interest going forward is computer vision for embedded/autonomous systems (perception and decision-making), rather than generative AI or NLP — the RNN notebook is included mainly for general understanding, not as a focus area.
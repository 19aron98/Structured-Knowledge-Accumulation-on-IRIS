# Structured Knowledge Accumulation (SKA) Model for Iris Dataset
## Overview
This repository contains an implementation of the Structured Knowledge Accumulation (SKA) model applied to the Iris dataset. The SKA framework offers an alternative to traditional backpropagation-based neural networks, using forward-only entropy minimization to optimize learning.

This project adapts an existing SKA model—initially designed for image classification—to work with the four-feature Iris dataset, demonstrating the feasibility of entropy-driven learning on structured data.

## What is SKA?
Structured Knowledge Accumulation (SKA) is a novel learning framework that replaces traditional gradient-based optimization with layer-wise entropy reduction. Instead of propagating errors backward, SKA aligns structured knowledge tensors (Z) with decision probability shifts (ΔD) to facilitate learning.

## Implementation Details
The implementation consists of the following key components:

1. Data Preparation
- The Iris dataset is loaded using scikit-learn.
- Features are normalized for stable training.
- Data is split into training and test sets (80/20 split).
  
2. Model Adaptation
- The original SKA model (designed for MNIST image classification) has been adapted.
- Input Layer: Adjusted to 4 neurons to match the 4 feature dimensions of the Iris dataset.
- Hidden Layers: Adjusted for the smaller dataset with sizes [24, 18, 12].
- Output Layer: Set to 3 neurons, one for each class (Iris-setosa, Iris-versicolor, Iris-virginica).
  
3. Training and Evaluation
- 300 forward steps are used for training without backpropagation.
- Entropy tracking per layer visualizes knowledge accumulation.
- Metrics such as Frobenius norms of knowledge tensors, cosine alignment, and decision probability evolution are monitored.
  
4. Comparison with Traditional Neural Networks
- A conventional feedforward neural network trained using backpropagation is implemented for comparison.
- The accuracy of both methods is compared to analyze learning efficiency and entropy dynamics.

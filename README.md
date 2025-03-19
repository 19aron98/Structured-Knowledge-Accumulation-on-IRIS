# Structured Knowledge Accumulation (SKA) Model for Iris Dataset
## Overview
This repository contains an implementation of the Structured Knowledge Accumulation (SKA) model applied to the Iris dataset. The SKA framework offers an alternative to traditional backpropagation-based neural networks, using forward-only entropy minimization to optimize learning.

This project adapts an existing SKA model—initially designed for image classification—to work with the four-feature Iris dataset, demonstrating the feasibility of entropy-driven learning on structured data.

## What is SKA?
Structured Knowledge Accumulation (SKA) is a novel learning framework that replaces traditional gradient-based optimization with layer-wise entropy reduction. Instead of propagating errors backward, SKA aligns structured knowledge tensors (Z) with decision probability shifts (ΔD) to facilitate learning.

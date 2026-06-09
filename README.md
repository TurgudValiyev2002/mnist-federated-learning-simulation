# MNIST-Style Federated Learning Simulation

## 1. Motivation

This lab simulates federated learning with the built-in scikit-learn digits dataset because real MNIST download may require internet. The research idea is FedAvg: clients train locally, then the server averages model parameters.

## 2. Project Goal

Build a small, reproducible AI research lab with clear outputs and honest limitations.

## 3. Dataset, Paper, Or Problem Description

Dataset: 8x8 handwritten digits from scikit-learn, used as an offline MNIST-style proxy.

## 4. Tools

Tools: Python, NumPy, pandas, scikit-learn, matplotlib.

## 5. Models Or Methods

Method: five clients, local SGD logistic regression, parameter averaging.

## 6. Hyperparameters When Relevant

Hyperparameters: 5 clients, 5 rounds, eta0=0.01, log-loss SGD, random_state=42.

## 7. Results

Results include round accuracy, client data sizes, and an accuracy curve.

## 8. Interpretation Of Results

Interpretation: accuracy should improve across rounds but this is still a simplified FL lab.

## 9. Conclusion

Conclusion: federated learning is a training protocol, not a magic model.

## 10. How To Run

```bash
pip install -r requirements.txt
python 1_*.py
```

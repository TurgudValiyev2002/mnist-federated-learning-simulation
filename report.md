# One-Page Report: MNIST-Style Federated Learning Simulation

## Motivation

This lab is part of Dear Turgud's June 2026 AI Research Lab. It focuses on a small but meaningful research workflow.

## Tools

Tools: Python, NumPy, pandas, scikit-learn, matplotlib.

## Dataset, Paper, Or Problem

Dataset: 8x8 handwritten digits from scikit-learn, used as an offline MNIST-style proxy.

## Model(s) Or Method(s)

Method: five clients, local SGD logistic regression, parameter averaging.

## Hyperparameters

Hyperparameters: 5 clients, 5 rounds, eta0=0.01, log-loss SGD, random_state=42.

## Results

Results include round accuracy, client data sizes, and an accuracy curve.

## Interpretation

Interpretation: accuracy should improve across rounds but this is still a simplified FL lab.

## Conclusion

Conclusion: federated learning is a training protocol, not a magic model.

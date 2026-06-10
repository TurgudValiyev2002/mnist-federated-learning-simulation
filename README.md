# MNIST-Style Federated Learning Simulation

## Motivation

Federated learning is useful when data is distributed across devices or institutions and should not be collected in one central location. This project simulates that idea with a small handwritten-digit dataset.

## Project Goal

We simulated federated learning with five clients. Each client trained locally, and the server averaged model parameters after each round.

## Dataset

We used the scikit-learn digits dataset as a small MNIST-style proxy. It contains 8x8 grayscale digit images represented as numeric features. This keeps the experiment runnable without downloading external data.

## Tools

Python, NumPy, pandas, scikit-learn, and matplotlib.

## Method

The data was split into train and test sets. The training data was divided across five clients. Each client trained an `SGDClassifier` locally, and the server averaged the learned coefficients and intercepts.

## Hyperparameters

- Clients: 5
- Federated rounds: 5
- Local model: logistic regression through `SGDClassifier(loss="log_loss")`
- Learning rate: `eta0=0.01`
- Test size: 20 percent
- Random seed: 42

## Results

| Round | Test Accuracy |
|---:|---:|
| 1 | 0.8972 |
| 2 | 0.9000 |
| 3 | 0.9083 |
| 4 | 0.9111 |
| 5 | 0.9056 |

Results are saved in `results/federated_round_metrics.csv`, `results/client_data_sizes.csv`, and `results/accuracy_by_round.png`.

## Interpretation

Accuracy improved from round 1 to round 4, then slightly decreased in round 5. This is realistic for a simple simulation: federated averaging can improve the global model, but the process is not automatically monotonic.

## Conclusion

The project demonstrates the main idea of federated learning: local training plus server aggregation. A stronger next version should test non-IID client data, more rounds, and client-level drift.

## How To Run

```bash
pip install -r requirements.txt
python 1_federated_digits_simulation.py
```

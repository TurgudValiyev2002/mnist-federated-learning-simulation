# Report: MNIST-Style Federated Learning Simulation

## Motivation

We simulated federated learning to understand how multiple clients can train locally while sharing only model updates.

## Dataset

The experiment used the scikit-learn digits dataset, an 8x8 handwritten-digit dataset that works as a small MNIST-style proxy.

## Method

Training data was split across five clients. Each client trained an SGD logistic model, and the server averaged model parameters for five rounds.

## Hyperparameters

We used 5 clients, 5 federated rounds, `SGDClassifier(loss="log_loss")`, learning rate `eta0=0.01`, and random seed 42.

## Results

Test accuracy moved from 0.8972 in round 1 to 0.9111 in round 4, then finished at 0.9056 in round 5.

## Interpretation

The global model improved overall, but the final small drop shows that federated training can fluctuate. More rounds and better local optimization would be needed for a stronger result.

## Conclusion

This project gives a simple working view of federated averaging. The next improvement should be non-IID client splits.

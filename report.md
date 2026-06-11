# One-Page Report: Real MNIST Federated Learning

## Motivation

We wanted a stronger federated learning experiment using real MNIST, not a small proxy dataset. The goal was to compare centralized training with federated learning under IID and non-IID client splits.

## Dataset

The experiment uses real MNIST with 60,000 training images and 10,000 test images. Each image is a 28x28 grayscale digit.

## Method

We trained a centralized SGD logistic classifier and federated versions with 2, 5, and 10 clients. Federated learning used 12 rounds. We compared IID and non-IID splits plus weighted FedAvg and uniform averaging.

## Results

The centralized baseline achieved 0.9051 accuracy. IID federated learning reached 0.9121 with 2 clients, 0.9097 with 5 clients, and 0.9027 with 10 clients. Non-IID federated learning was weaker: 0.7192 with 2 clients, 0.5660 with 5 clients, and 0.6312 with 10 clients.

## Interpretation

IID FL can match centralized training because each client sees all digit classes. Non-IID FL is much harder because clients train on skewed label distributions, causing local updates to conflict.

## Conclusion

The project gives a realistic MNIST federated learning comparison. The main conclusion is that data heterogeneity matters more than the averaging formula when clients have equal sample sizes.

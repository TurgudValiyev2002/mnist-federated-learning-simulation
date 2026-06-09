from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RESULTS = Path("results")
RNG = np.random.default_rng(42)

def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    x, y = load_digits(return_X_y=True)
    x = StandardScaler().fit_transform(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)
    clients = np.array_split(RNG.permutation(len(x_train)), 5)
    classes = np.unique(y)
    global_coef = None
    global_intercept = None
    rows = []
    for rnd in range(1, 6):
        coefs, intercepts, weights = [], [], []
        for cid, idx in enumerate(clients):
            clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=100 + rnd + cid, learning_rate="constant", eta0=0.01)
            clf.partial_fit(x_train[idx], y_train[idx], classes=classes)
            if global_coef is not None:
                clf.coef_ = 0.5 * clf.coef_ + 0.5 * global_coef
                clf.intercept_ = 0.5 * clf.intercept_ + 0.5 * global_intercept
            coefs.append(clf.coef_)
            intercepts.append(clf.intercept_)
            weights.append(len(idx))
        global_coef = np.average(coefs, axis=0, weights=weights)
        global_intercept = np.average(intercepts, axis=0, weights=weights)
        pred = x_test @ global_coef.T + global_intercept
        acc = accuracy_score(y_test, pred.argmax(axis=1))
        rows.append({"round": rnd, "test_accuracy": round(acc, 4)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "federated_round_metrics.csv", index=False)
    pd.DataFrame({"client": range(5), "train_samples": [len(c) for c in clients]}).to_csv(RESULTS / "client_data_sizes.csv", index=False)
    plt.figure(figsize=(6, 4))
    plt.plot(metrics["round"], metrics["test_accuracy"], marker="o")
    plt.ylim(0, 1)
    plt.xlabel("Federated round")
    plt.ylabel("Test accuracy")
    plt.title("Federated Digits Simulation")
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_by_round.png", dpi=160)
    print(metrics.to_string(index=False))

if __name__ == "__main__":
    main()

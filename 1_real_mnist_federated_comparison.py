from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
ASSETS = ROOT / "assets"
BASE_URLS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist",
    "https://github.com/mkolod/MNIST/raw/master",
]
FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
SEED = 42


def download_mnist() -> None:
    DATA.mkdir(exist_ok=True)
    for filename in FILES.values():
        path = DATA / filename
        if path.exists():
            continue
        last_error = None
        for base_url in BASE_URLS:
            try:
                url = f"{base_url}/{filename}"
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, path)
                last_error = None
                break
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise last_error


def read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        _, n, rows, cols = struct.unpack(">IIII", handle.read(16))
        data = np.frombuffer(handle.read(), dtype=np.uint8)
    return data.reshape(n, rows * cols).astype(np.float32) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        _, _ = struct.unpack(">II", handle.read(8))
        labels = np.frombuffer(handle.read(), dtype=np.uint8)
    return labels.astype(np.int64)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    download_mnist()
    return (
        read_idx_images(DATA / FILES["train_images"]),
        read_idx_labels(DATA / FILES["train_labels"]),
        read_idx_images(DATA / FILES["test_images"]),
        read_idx_labels(DATA / FILES["test_labels"]),
    )


def centralize_train(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="constant", eta0=0.02, max_iter=15, random_state=SEED, n_jobs=-1)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    return pd.DataFrame(
        [
            {
                "model": "centralized_sgd_logistic",
                "train_images": len(x_train),
                "test_images": len(x_test),
                "accuracy": round(accuracy_score(y_test, pred), 4),
            }
        ]
    )


def make_iid_clients(y_train: np.ndarray, n_clients: int) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED + n_clients)
    return [idx for idx in np.array_split(rng.permutation(len(y_train)), n_clients)]


def make_noniid_clients(y_train: np.ndarray, n_clients: int, shards_per_client: int = 2) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED + n_clients)
    sorted_idx = np.argsort(y_train)
    shards = np.array_split(sorted_idx, n_clients * shards_per_client)
    shard_order = rng.permutation(len(shards))
    clients = []
    for cid in range(n_clients):
        chosen = shard_order[cid * shards_per_client : (cid + 1) * shards_per_client]
        clients.append(np.concatenate([shards[s] for s in chosen]))
    return clients


def fit_local_model(x: np.ndarray, y: np.ndarray, classes: np.ndarray, coef: np.ndarray | None, intercept: np.ndarray | None, seed: int):
    clf = SGDClassifier(loss="log_loss", alpha=1e-4, learning_rate="constant", eta0=0.02, max_iter=1, tol=None, random_state=seed)
    clf.partial_fit(x, y, classes=classes)
    if coef is not None:
        clf.coef_ = 0.65 * clf.coef_ + 0.35 * coef
        clf.intercept_ = 0.65 * clf.intercept_ + 0.35 * intercept
    return clf.coef_, clf.intercept_


def aggregate(coefs: list[np.ndarray], intercepts: list[np.ndarray], sizes: list[int], method: str):
    if method == "weighted_fedavg":
        weights = np.array(sizes, dtype=np.float64)
    elif method == "uniform_average":
        weights = np.ones(len(sizes), dtype=np.float64)
    else:
        raise ValueError(method)
    coef = np.average(coefs, axis=0, weights=weights)
    intercept = np.average(intercepts, axis=0, weights=weights)
    return coef, intercept


def evaluate_linear(coef: np.ndarray, intercept: np.ndarray, x_test: np.ndarray, y_test: np.ndarray) -> float:
    pred = (x_test @ coef.T + intercept).argmax(axis=1)
    return accuracy_score(y_test, pred)


def run_federated(x_train, y_train, x_test, y_test, n_clients: int, split_type: str, agg_method: str):
    clients = make_iid_clients(y_train, n_clients) if split_type == "iid" else make_noniid_clients(y_train, n_clients)
    classes = np.arange(10)
    coef = None
    intercept = None
    rows = []
    rounds = 12
    for rnd in range(1, rounds + 1):
        local_coefs, local_intercepts, sizes = [], [], []
        for cid, idx in enumerate(clients):
            c, b = fit_local_model(x_train[idx], y_train[idx], classes, coef, intercept, seed=SEED + rnd * 31 + cid)
            local_coefs.append(c)
            local_intercepts.append(b)
            sizes.append(len(idx))
        coef, intercept = aggregate(local_coefs, local_intercepts, sizes, agg_method)
        rows.append(
            {
                "clients": n_clients,
                "split": split_type,
                "aggregation": agg_method,
                "round": rnd,
                "test_accuracy": round(evaluate_linear(coef, intercept, x_test, y_test), 4),
            }
        )
    client_rows = []
    for cid, idx in enumerate(clients):
        counts = np.bincount(y_train[idx], minlength=10)
        client_rows.append(
            {
                "clients": n_clients,
                "split": split_type,
                "client": cid,
                "train_samples": len(idx),
                "dominant_label": int(counts.argmax()),
                "unique_labels": int((counts > 0).sum()),
                **{f"label_{i}": int(counts[i]) for i in range(10)},
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(client_rows)


def plot_results(metrics: pd.DataFrame, centralized: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    for (clients, split, agg), group in metrics.groupby(["clients", "split", "aggregation"]):
        if clients in [5, 10] and agg == "weighted_fedavg":
            plt.plot(group["round"], group["test_accuracy"], marker="o", label=f"{clients} clients {split}")
    plt.axhline(float(centralized["accuracy"].iloc[0]), color="black", linestyle="--", label="centralized")
    plt.ylim(0.5, 1.0)
    plt.xlabel("Federated round")
    plt.ylabel("Test accuracy")
    plt.title("Real MNIST FedAvg: IID vs Non-IID")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "accuracy_by_round.png", dpi=180)
    plt.close()

    final = metrics.sort_values("round").groupby(["clients", "split", "aggregation"]).tail(1)
    labels = final.apply(lambda r: f"{int(r['clients'])}-{r['split']}-{r['aggregation'].replace('_', ' ')}", axis=1)
    plt.figure(figsize=(10, 5))
    plt.bar(labels, final["test_accuracy"], color="#3d6fb6")
    plt.axhline(float(centralized["accuracy"].iloc[0]), color="black", linestyle="--", label="centralized")
    plt.ylim(0.5, 1.0)
    plt.ylabel("Final test accuracy")
    plt.title("Final Accuracy Across FL Settings")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS / "final_accuracy_comparison.png", dpi=180)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    boxes = [
        ("Real MNIST\n60k train / 10k test", 0.15),
        ("Centralized\nbaseline", 0.39),
        ("FL clients\n2, 5, 10", 0.63),
        ("IID/non-IID +\naggregation comparison", 0.86),
    ]
    for text, xpos in boxes:
        ax.text(xpos, 0.55, text, ha="center", va="center", fontsize=12, bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef6ff", edgecolor="#336699"))
    for start, end in zip(boxes[:-1], boxes[1:]):
        ax.annotate("", xy=(end[1] - 0.11, 0.55), xytext=(start[1] + 0.11, 0.55), arrowprops=dict(arrowstyle="->", lw=2))
    ax.set_title("Real MNIST centralized vs federated learning workflow", fontsize=15)
    fig.tight_layout()
    fig.savefig(ASSETS / "readme_project_overview.png", dpi=180)
    plt.close(fig)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    ASSETS.mkdir(exist_ok=True)
    x_train, y_train, x_test, y_test = load_mnist()
    centralized = centralize_train(x_train, y_train, x_test, y_test)
    centralized.to_csv(RESULTS / "centralized_baseline.csv", index=False)
    all_metrics = []
    all_clients = []
    for n_clients in [2, 5, 10]:
        for split in ["iid", "non_iid"]:
            for agg in ["weighted_fedavg", "uniform_average"]:
                metrics, clients = run_federated(x_train, y_train, x_test, y_test, n_clients, split, agg)
                all_metrics.append(metrics)
                all_clients.append(clients)
    metrics_df = pd.concat(all_metrics, ignore_index=True)
    clients_df = pd.concat(all_clients, ignore_index=True).drop_duplicates(subset=["clients", "split", "client"])
    metrics_df.to_csv(RESULTS / "federated_round_metrics.csv", index=False)
    clients_df.to_csv(RESULTS / "client_label_distribution.csv", index=False)
    final = metrics_df.sort_values("round").groupby(["clients", "split", "aggregation"]).tail(1)
    final.to_csv(RESULTS / "final_comparison.csv", index=False)
    pd.DataFrame(
        [
            {"metric": "train_images", "value": len(x_train)},
            {"metric": "test_images", "value": len(x_test)},
            {"metric": "rounds", "value": 12},
            {"metric": "client_counts", "value": "2,5,10"},
            {"metric": "aggregations", "value": "weighted_fedavg,uniform_average"},
        ]
    ).to_csv(RESULTS / "experiment_setup.csv", index=False)
    plot_results(metrics_df, centralized)
    print(centralized.to_string(index=False))
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()

import struct
import sys

import matplotlib.pyplot as plt
import numpy as np


def read_pod(f, fmt):
    sz = struct.calcsize(fmt)
    data = f.read(sz)
    if len(data) < sz:
        raise EOFError
    return struct.unpack(fmt, data)[0]


def read_table(f):
    sz = read_pod(f, "Q")
    table = []
    for _ in range(sz):
        score = read_pod(f, "i")
        recall_size = read_pod(f, "Q")
        recalls = []
        for _ in range(recall_size):
            ef = read_pod(f, "i")
            recall = read_pod(f, "f")
            recalls.append((ef, recall))
        table.append((score, recalls))
    return table


def deserialize(filename):
    with open(filename, "rb") as f:
        ef_recall_estimators = read_table(f)
        expected_recall = read_pod(f, "f")
        wae = read_pod(f, "f")

        n_dep = read_pod(f, "Q")
        dep_tables = []
        for _ in range(n_dep):
            dep_tables.append(read_table(f))

        n_thresh = read_pod(f, "Q")
        dep_centers = []
        for _ in range(n_thresh):
            dep_centers.append(read_pod(f, "f"))

    return ef_recall_estimators, expected_recall, wae, dep_tables, dep_centers


def table_to_dict(table):
    return {score: ef_recalls for score, ef_recalls in table}


def lookup_ef(table_dict, score, expected_recall):
    clamped = max(0, min(100, int(score)))
    if clamped not in table_dict:
        return np.nan
    ef_recalls = table_dict[clamped]
    for ef, recall in ef_recalls:
        if recall >= expected_recall:
            return ef
    return ef_recalls[-1][0]


def estimate_ef2(tables_dicts, dep_centers, score, d_ep, expected_recall):
    n_centers = len(dep_centers)
    if n_centers == 1:
        return lookup_ef(tables_dicts[0], score, expected_recall)

    if d_ep <= dep_centers[0]:
        return lookup_ef(tables_dicts[0], score, expected_recall)
    if d_ep >= dep_centers[-1]:
        return lookup_ef(tables_dicts[-1], score, expected_recall)

    idx = 0
    while idx < n_centers - 1 and d_ep > dep_centers[idx + 1]:
        idx += 1

    c0 = dep_centers[idx]
    c1 = dep_centers[idx + 1]
    w = (d_ep - c0) / (c1 - c0) if c1 != c0 else 0

    ef0 = lookup_ef(tables_dicts[idx], score, expected_recall)
    ef1 = lookup_ef(tables_dicts[idx + 1], score, expected_recall)

    if np.isnan(ef0) or np.isnan(ef1):
        return np.nan

    return ef0 * (1.0 - w) + ef1 * w


def plot_heatmap(filename, output_png):
    _, expected_recall, _, dep_tables, dep_centers = deserialize(filename)

    if not dep_tables:
        print("No dependency tables found.")
        return

    tables_dicts = [table_to_dict(t) for t in dep_tables]

    scores = np.linspace(0, 100, 101)

    d_ep_min = dep_centers[0] - (dep_centers[-1] - dep_centers[0]) * 0.1
    d_ep_max = dep_centers[-1] + (dep_centers[-1] - dep_centers[0]) * 0.1

    if d_ep_min == d_ep_max:
        d_ep_min -= 1.0
        d_ep_max += 1.0

    d_eps = np.linspace(d_ep_min, d_ep_max, 100)

    X, Y = np.meshgrid(scores, d_eps)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = estimate_ef2(
                tables_dicts, dep_centers, X[i, j], Y[i, j], expected_recall
            )

    plt.figure(figsize=(10, 8))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="white")
    plt.contourf(X, Y, Z, levels=50, cmap=cmap)
    plt.colorbar(label="Estimated ef")

    for c in dep_centers:
        plt.axhline(
            y=c,
            color="r",
            linestyle="--",
            alpha=0.5,
            label="Center" if c == dep_centers[0] else "",
        )

    plt.xlabel("Coefficient of Variation (CV Score)")
    plt.ylabel("Revisit-Order (Convergence Score)")
    plt.title(
        f"Estimated ef mapped on CV / Convergence Coordinate Plane\nExpected Recall: {expected_recall}"
    )
    plt.legend()
    plt.savefig(output_png)
    print(f"Heatmap saved to {output_png}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <ef_adaptor.bin> <output.png>")
        sys.exit(1)
    plot_heatmap(sys.argv[1], sys.argv[2])

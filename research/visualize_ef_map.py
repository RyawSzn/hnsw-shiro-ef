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

    ef0 = lookup_ef(tables_dicts[idx], score, expected_recall)
    ef1 = lookup_ef(tables_dicts[idx + 1], score, expected_recall)

    if np.isnan(ef0) and np.isnan(ef1):
        return np.nan
    if np.isnan(ef0):
        return ef1
    if np.isnan(ef1):
        return ef0

    # Smooth linear interpolation between the two nearest CV tables
    dist_total = c1 - c0
    if dist_total == 0:
        return ef0
    w1 = (d_ep - c0) / dist_total
    w0 = 1.0 - w1
    return ef0 * w0 + ef1 * w1


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

    d_eps = np.linspace(
        d_ep_min, d_ep_max, 300
    )  # Increased Y resolution for smoother interpolation

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
    plt.pcolormesh(
        X, Y, Z, cmap=cmap, shading="gouraud"
    )  # Changed shading to gouraud for smooth gradients
    plt.colorbar(label="Estimated ef")

    # Only plot center lines if there aren't too many (e.g. don't clutter if n_cv_tables=0 creates 101 centers)
    if len(dep_centers) <= 20:
        for c in dep_centers:
            plt.axhline(
                y=c,
                color="r",
                linestyle="--",
                alpha=0.5,
                label="Center" if c == dep_centers[0] else "",
            )
    else:
        print(
            f"Skipping drawing {len(dep_centers)} center lines to prevent cluttering the heatmap."
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
    import os
    import sys
    from pathlib import Path

    # Search specifically in EXPERIMENTS_ROOT/estimation_table
    base_dir = os.environ.get("EXPERIMENTS_ROOT", ".")
    target_dir = Path(base_dir) / "estimation_table_ada"

    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Directory not found: {target_dir}")
        print(
            "Please ensure EXPERIMENTS_ROOT is set correctly or the directory exists."
        )
        sys.exit(1)

    bin_files = []
    for p in target_dir.rglob("*.bin"):
        if "ef_adaptor" in p.name or "-ef.bin" in p.name or "ef-recomp.bin" in p.name:
            if p not in bin_files:
                bin_files.append(p)

    if not bin_files:
        print(f"No estimation table (.bin) files found in {target_dir}.")
        sys.exit(1)

    print(f"Available estimation tables in {target_dir}:")
    for i, f in enumerate(bin_files):
        print(f"[{i}] {f.name}")
    print("[all] Process all datasets")

    idx_str = (
        input("\nEnter the number of the dataset to visualize (or 'all'): ")
        .strip()
        .lower()
    )

    files_to_process = []
    if idx_str == "all":
        files_to_process = bin_files
    else:
        try:
            idx = int(idx_str)
            files_to_process = [bin_files[idx]]
        except (ValueError, IndexError):
            print("Invalid selection. Exiting.")
            sys.exit(1)

    # Ask about replacement mode
    replace_str = (
        input("Overwrite existing images if they exist? (y/n) [n]: ").strip().lower()
    )
    replace_mode = replace_str == "y" or replace_str == "yes"

    # Prepare output path directory
    out_dir = Path("research/img/heatmap")
    out_dir.mkdir(parents=True, exist_ok=True)

    for selected_file in files_to_process:
        # Extract dataset name (heuristic based on filename pattern)
        name_parts = selected_file.name.split("-ef_adaptor")
        if len(name_parts) > 1:
            dataset_name = name_parts[0]
        else:
            # fallback
            dataset_name = selected_file.stem.replace("-ef", "").replace("-recomp", "")

        base_name = f"output_heatmap_{dataset_name}"
        out_png = out_dir / f"{base_name}.png"

        if not replace_mode:
            counter = 1
            while out_png.exists():
                out_png = out_dir / f"{base_name}_{counter}.png"
                counter += 1

        print(f"\nProcessing {selected_file} -> {out_png}\n")
        try:
            plot_heatmap(str(selected_file), str(out_png))
        except Exception as e:
            print(f"Error processing {selected_file.name}: {e}")

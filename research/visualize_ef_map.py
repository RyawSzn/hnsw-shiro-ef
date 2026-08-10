import argparse
import math
import os
import struct
import sys
from pathlib import Path

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

        easy_recall = None
        conv_p20 = None
        try:
            easy_recall = read_pod(f, "f")
            conv_p20 = read_pod(f, "f")
        except EOFError:
            pass

    return (
        ef_recall_estimators,
        expected_recall,
        wae,
        dep_tables,
        dep_centers,
        easy_recall,
        conv_p20,
    )


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

    dist_total = c1 - c0
    if dist_total == 0:
        return ef0
    w1 = (d_ep - c0) / dist_total
    w0 = 1.0 - w1
    return ef0 * w0 + ef1 * w1


def compute_heatmap_data(filename, use_dual_recall):
    _, hard_recall, _, dep_tables, dep_centers, easy_recall, conv_p20 = deserialize(
        filename
    )

    if not dep_tables:
        raise ValueError("No dependency tables found.")

    tables_dicts = [table_to_dict(t) for t in dep_tables]
    scores = np.linspace(0, 100, 101)

    d_ep_min = dep_centers[0] - (dep_centers[-1] - dep_centers[0]) * 0.1
    d_ep_max = dep_centers[-1] + (dep_centers[-1] - dep_centers[0]) * 0.1

    if d_ep_min == d_ep_max:
        d_ep_min -= 1.0
        d_ep_max += 1.0

    d_eps = np.linspace(d_ep_min, d_ep_max, 300)

    X, Y = np.meshgrid(scores, d_eps)
    Z = np.zeros_like(X)

    dual = use_dual_recall and easy_recall is not None and conv_p20 is not None

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if dual:
                target = hard_recall if Y[i, j] <= conv_p20 else easy_recall
            else:
                target = hard_recall
            Z[i, j] = estimate_ef2(tables_dicts, dep_centers, X[i, j], Y[i, j], target)

    return X, Y, Z, dep_centers, hard_recall, easy_recall, conv_p20


def dataset_name_from_path(p):
    name_parts = p.name.split("-ef_adaptor")
    if len(name_parts) > 1:
        return name_parts[0]
    return (
        p.stem.replace("-ef_adapter", "")
        .replace("-ef_adaptor", "")
        .replace("-ef", "")
        .replace("-recomp", "")
    )


def render_all(files_to_process, out_png, use_dual_recall):
    num_ds = len(files_to_process)
    ncols = min(num_ds, 3)
    nrows = math.ceil(num_ds / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.9 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for i in range(num_ds, len(axes_flat)):
        axes_flat[i].set_visible(False)

    cmap = plt.get_cmap("viridis_r").copy()
    cmap.set_bad(color="lightgray")

    has_centers = False
    has_p20 = False
    z_global_min, z_global_max = np.inf, -np.inf

    datasets = []
    for selected_file in files_to_process:
        ds_name = dataset_name_from_path(selected_file)
        print(f"Computing {selected_file.name} ...")
        try:
            X, Y, Z, dep_centers, hard_recall, easy_recall, conv_p20 = (
                compute_heatmap_data(str(selected_file), use_dual_recall)
            )
            datasets.append(
                (ds_name, X, Y, Z, dep_centers, hard_recall, easy_recall, conv_p20)
            )
            valid = Z[~np.isnan(Z)]
            if valid.size:
                z_global_min = min(z_global_min, valid.min())
                z_global_max = max(z_global_max, valid.max())
        except Exception as e:
            print(f"  Error: {e} — skipping.")
            datasets.append(None)

    if z_global_min == np.inf:
        print("No valid data to plot.")
        return

    mesh_ref = None
    for idx, entry in enumerate(datasets):
        ax = axes_flat[idx]
        if entry is None:
            ax.set_visible(False)
            continue

        ds_name, X, Y, Z, dep_centers, hard_recall, easy_recall, conv_p20 = entry
        Z_masked = np.ma.masked_invalid(Z)

        mesh = ax.pcolormesh(
            X,
            Y,
            Z_masked,
            cmap=cmap,
            shading="gouraud",
            vmin=z_global_min,
            vmax=z_global_max,
        )
        mesh_ref = mesh

        if len(dep_centers) <= 20:
            has_centers = True
            for c in dep_centers:
                ax.axhline(y=c, color="r", linestyle="--", linewidth=0.8, alpha=0.5)

        ax.set_title(ds_name, fontsize=16)
        ax.set_box_aspect(1)
        ax.set_xlabel("CV Score", fontsize=12)
        ax.set_ylabel("Convergence Bucket", fontsize=12)
        ax.tick_params(axis="x", labelrotation=90, labelsize=11)
        ax.tick_params(axis="y", labelsize=11)

    if mesh_ref is not None:
        fig.colorbar(
            mesh_ref,
            ax=axes_flat[:num_ds].tolist(),
            label="Estimated ef",
            shrink=0.8,
            aspect=30,
        )

    hard_recalls = [e[5] for e in datasets if e is not None]
    easy_recalls = [e[6] for e in datasets if e is not None and e[6] is not None]
    hard_str = (
        f"{hard_recalls[0]:.2f}"
        if hard_recalls and all(r == hard_recalls[0] for r in hard_recalls)
        else "varies"
    )

    if use_dual_recall and easy_recalls:
        easy_str = (
            f"{easy_recalls[0]:.2f}"
            if all(r == easy_recalls[0] for r in easy_recalls)
            else "varies"
        )
        recall_label = f"hard={hard_str}, easy={easy_str}"
    else:
        recall_label = f"recall={hard_str}"

    fig.suptitle(
        f"Estimated ef — CV / Convergence Coordinate Plane  ({recall_label})",
        fontsize=15,
    )

    from matplotlib.lines import Line2D

    legend_handles = []
    if has_centers:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="r",
                linestyle="--",
                linewidth=0.8,
                alpha=0.5,
                label="Convergence bucket center",
            )
        )
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=len(legend_handles),
            fontsize=11,
            frameon=True,
            fancybox=False,
            edgecolor="black",
            columnspacing=0.8,
            handletextpad=0.3,
        )

    fig.align_xlabels()
    plt.savefig(out_png, dpi=300)
    print(f"\nHeatmap grid saved to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EF estimation heatmaps.")
    parser.add_argument(
        "--dual-recall",
        action="store_true",
        help="Use dual-recall mode: hard region (conv <= P20) uses the baked-in hard recall, "
        "easy region uses the baked-in easy recall. "
        "Draws the P20 boundary as a grey dotted line. "
        "Default: single recall (baked-in hard recall everywhere).",
    )
    args = parser.parse_args()

    base_dir = os.environ.get("EXPERIMENTS_ROOT", ".")
    target_dir = Path(base_dir) / "estimation_table"

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
        print(f"  [{i}] {f.name}")

    user_input = input(
        "\nEnter dataset numbers to visualize (comma-separated), or press Enter for all: "
    ).strip()

    files_to_process = []
    if not user_input or user_input.lower() == "all":
        files_to_process = bin_files
    else:
        selected_indices = []
        for p in user_input.split(","):
            p = p.strip()
            if p.isdigit():
                idx = int(p)
                if 0 <= idx < len(bin_files):
                    selected_indices.append(idx)
                else:
                    print(f"Warning: Index {idx} out of bounds, skipping.")
            else:
                print(f"Warning: Invalid input '{p}', skipping.")

        files_to_process = [bin_files[i] for i in dict.fromkeys(selected_indices)]

    if not files_to_process:
        print("No datasets selected. Exiting.")
        sys.exit(1)

    base_suffix = "dual" if args.dual_recall else "single"
    out_dir = Path("research/img/heatmap")
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(files_to_process) == 1:
        ds_name = dataset_name_from_path(files_to_process[0])
        base_name = f"output_heatmap_{ds_name}_{base_suffix}"
    else:
        base_name = f"output_heatmap_grid_{len(files_to_process)}ds_{base_suffix}"

    out_png = out_dir / f"{base_name}.png"
    counter = 1
    while out_png.exists():
        out_png = out_dir / f"{base_name}_{counter}.png"
        counter += 1

    render_all(files_to_process, str(out_png), args.dual_recall)

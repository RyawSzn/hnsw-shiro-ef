import math
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator


def parse_log(filepath):
    try:
        with open(filepath, "r") as f:
            lines = f.readlines()
    except:
        return {}

    datasets = {}
    current_dataset = None
    current_block = None
    current_ef = None
    current_run = {}

    for line in lines:
        line = line.strip()

        m_ds = re.match(r"Dataset: (.*)", line)
        if m_ds:
            current_dataset = m_ds.group(1)
            if current_dataset not in datasets:
                datasets[current_dataset] = {"our_method": {}, "baseline": {}}
            current_block = "our_method"
            continue

        m_patience = re.match(r"Search with Patience in Proximity", line)
        if m_patience:
            current_block = "patience"
            continue

        if line == "Experiment finished":
            if current_block == "patience":
                current_block = "baseline"
            continue

        m_ef = re.match(r"ef: (\d+)", line)
        if m_ef and current_block == "baseline":
            current_ef = int(m_ef.group(1))
            current_run = {}
            continue

        m_st = re.match(r"Search times: (.*)", line)
        if m_st:
            times_str = m_st.group(1)
            times = [
                int(x.strip().replace("ms", ""))
                for x in times_str.split(",")
                if x.strip()
            ]
            times.sort()
            current_run["times"] = times

            if current_block == "our_method" and current_dataset:
                datasets[current_dataset]["our_method"] = current_run.copy()
            elif current_block == "baseline" and current_dataset and current_ef:
                datasets[current_dataset]["baseline"][current_ef] = current_run.copy()
            current_run = {}
            continue

        m_avg = re.match(r"Average Recall: (.*)", line)
        if m_avg:
            current_run["avg"] = float(m_avg.group(1))

        m_p05 = re.match(r"5th percentile recall: (.*)", line)
        if m_p05:
            current_run["p05"] = float(m_p05.group(1))

        m_p01 = re.match(r"1st percentile recall: (.*)", line)
        if m_p01:
            current_run["p01"] = float(m_p01.group(1))

    return datasets


shiro = parse_log("research/log/output_shiro_new0.log")
ada = parse_log("research/log/output_ada.log")
progress = parse_log("research/log/output_shiro_progress.log")

datasets_keys = sorted(
    list(set(list(shiro.keys()) + list(ada.keys()) + list(progress.keys())))
)
num_ds = len(datasets_keys)

global_all_recalls = []
for ds_name in datasets_keys:
    if "baseline" in shiro.get(ds_name, {}):
        for ef, s_log in shiro[ds_name]["baseline"].items():
            global_all_recalls.extend([s_log["avg"], s_log["p05"], s_log["p01"]])
    if "baseline" in ada.get(ds_name, {}):
        for ef, a_log in ada[ds_name]["baseline"].items():
            global_all_recalls.extend([a_log["avg"], a_log["p05"], a_log["p01"]])
    if "our_method" in shiro.get(ds_name, {}):
        om = shiro[ds_name]["our_method"]
        if "avg" in om:
            global_all_recalls.append(om["avg"])
        if "p05" in om:
            global_all_recalls.append(om["p05"])
        if "p01" in om:
            global_all_recalls.append(om["p01"])
    if "our_method" in ada.get(ds_name, {}):
        om = ada[ds_name]["our_method"]
        if "avg" in om:
            global_all_recalls.append(om["avg"])
        if "p05" in om:
            global_all_recalls.append(om["p05"])
        if "p01" in om:
            global_all_recalls.append(om["p01"])

import math

global_min_y = (
    math.floor(min(global_all_recalls) / 0.05) * 0.05 if global_all_recalls else 0.70
)

ncols = 3
nrows = math.ceil(num_ds / ncols)
fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6.5 * nrows))
axes_flat = axes.flatten() if num_ds > 1 else [axes]
for i in range(num_ds, len(axes_flat)):
    axes_flat[i].set_visible(False)

for idx, ds_name in enumerate(datasets_keys):
    ax = axes_flat[idx]

    # 1. First Pass: Compute the relative hardware scaling factor (Ada -> Shiro)
    common_efs = set(shiro.get(ds_name, {}).get("baseline", {}).keys()).intersection(
        set(ada.get(ds_name, {}).get("baseline", {}).keys())
    )

    total_s_time = 0
    total_a_time = 0
    for ef in common_efs:
        s_t = shiro[ds_name]["baseline"][ef]["times"]
        a_t = ada[ds_name]["baseline"][ef]["times"]
        total_s_time += s_t[1] if len(s_t) > 1 else s_t[0]
        total_a_time += a_t[1] if len(a_t) > 1 else a_t[0]

    # Gamma scales Ada's clock speed to match Shiro's hardware environment
    gamma = total_s_time / total_a_time if total_a_time > 0 else 1.0

    # 2. Second Pass: Extract data points using Shiro as primary, falling back to scaled Ada
    bl_efs = sorted(
        list(
            set(shiro.get(ds_name, {}).get("baseline", {}).keys()).union(
                set(ada.get(ds_name, {}).get("baseline", {}).keys())
            )
        )
    )

    bl_times_med = []
    bl_avg = []
    bl_p05 = []
    bl_p01 = []

    for ef in bl_efs:
        s_log = shiro.get(ds_name, {}).get("baseline", {}).get(ef, {})
        a_log = ada.get(ds_name, {}).get("baseline", {}).get(ef, {})

        if s_log:
            # Shiro is our baseline hardware anchor
            times = s_log["times"]
            med_val = times[1] if len(times) > 1 else times[0]
            bl_times_med.append(med_val / 1000.0)
            bl_avg.append(s_log["avg"])
            bl_p05.append(s_log["p05"])
            bl_p01.append(s_log["p01"])
        elif a_log:
            # Shiro doesn't have this ef; use Ada's recall but NORMALIZE its latency
            times = a_log["times"]
            med_val = times[1] if len(times) > 1 else times[0]
            bl_times_med.append((med_val * gamma) / 1000.0)  # Apply hardware correction
            bl_avg.append(a_log["avg"])
            bl_p05.append(a_log["p05"])
            bl_p01.append(a_log["p01"])

    # 3. Plot the synchronized curves
    ax.plot(
        bl_times_med, bl_avg, "-", color="tab:blue", label="Baseline - Avg", linewidth=2
    )
    ax.plot(
        bl_times_med,
        bl_p05,
        "-",
        color="tab:green",
        label=r"Baseline - $\pi_{0.05}$",
        linewidth=2,
    )
    ax.plot(
        bl_times_med,
        bl_p01,
        "-",
        color="tab:orange",
        label=r"Baseline - $\pi_{0.01}$",
        linewidth=2,
    )

    ax.fill_between(bl_times_med, bl_p01, bl_avg, color="silver", alpha=0.15, zorder=1)

    if "our_method" in shiro.get(ds_name, {}):
        s_data = shiro[ds_name]["our_method"]
        if s_data.get("times"):
            s_time = min(s_data["times"]) / 1000.0
            ax.scatter(
                [s_time],
                [s_data["avg"]],
                marker="*",
                s=350,
                color="blue",
                edgecolor="black",
                label=r"Shiro - Avg",
                zorder=5,
            )
            ax.scatter(
                [s_time],
                [s_data["p05"]],
                marker="*",
                s=350,
                color="green",
                edgecolor="black",
                label=r"Shiro - $\pi_{0.05}$",
                zorder=5,
            )
            ax.scatter(
                [s_time],
                [s_data["p01"]],
                marker="*",
                s=350,
                color="orange",
                edgecolor="black",
                label=r"Shiro - $\pi_{0.01}$",
                zorder=5,
            )

    if "our_method" in ada.get(ds_name, {}):
        a_data = ada[ds_name]["our_method"]
        if a_data.get("times"):
            a_time = (
                a_data["times"][1] if len(a_data["times"]) > 1 else a_data["times"][0]
            ) / 1000.0
            ax.scatter(
                [a_time],
                [a_data["avg"]],
                marker="X",
                s=250,
                color="blue",
                edgecolor="black",
                label=r"Ada - Avg",
                zorder=5,
            )
            ax.scatter(
                [a_time],
                [a_data["p05"]],
                marker="X",
                s=250,
                color="green",
                edgecolor="black",
                label=r"Ada - $\pi_{0.05}$",
                zorder=5,
            )
            ax.scatter(
                [a_time],
                [a_data["p01"]],
                marker="X",
                s=250,
                color="orange",
                edgecolor="black",
                label=r"Ada - $\pi_{0.01}$",
                zorder=5,
            )

    if "our_method" in progress.get(ds_name, {}):
        p_data = progress[ds_name]["our_method"]
        if p_data.get("times"):
            p_time = min(p_data["times"]) / 1000.0
            for val in [p_data.get("avg"), p_data.get("p05"), p_data.get("p01")]:
                if val is not None:
                    ax.scatter(
                        [p_time],
                        [val],
                        marker="^",
                        s=220,
                        color="grey",
                        edgecolor="black",
                        zorder=2,
                    )

    max_ef = max(bl_efs) if bl_efs else "Unknown"
    ax.set_title(f"{ds_name}\nObserved ef_max = {max_ef}", fontsize=14)

    ax.set_xlabel("Latency (s)", fontsize=12)

    ax.set_ylabel("Recall@100", fontsize=12)

    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    ax.set_ylim(0.70, 1.01)

    max_time = max(bl_times_med) if bl_times_med else 10
    if "our_method" in shiro.get(ds_name, {}):
        if shiro[ds_name]["our_method"].get("times"):
            max_time = max(
                max_time, min(shiro[ds_name]["our_method"]["times"]) / 1000.0
            )
    if "our_method" in ada.get(ds_name, {}):
        if ada[ds_name]["our_method"].get("times"):
            max_time = max(max_time, min(ada[ds_name]["our_method"]["times"]) / 1000.0)

    ax.set_xlim(0, max_time * 1.1)

    # ADD TARGET RECALL LINE
    ax.axhline(
        y=0.95, color="tab:red", linestyle="-.", alpha=0.8, linewidth=1.5, zorder=1
    )

    # ADD TEXT STATING IT'S THE TARGET RECALL
    x_min, x_max = ax.get_xlim()
    offset = (x_max - x_min) * 0.02
    ax.text(
        x_min + offset,
        0.952,
        "Target Recall",
        color="tab:red",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=6,
    )

    ax.grid(True, linestyle="--", alpha=0.6)


plt.tight_layout(rect=(0, 0.03, 1, 1))

legend_handles = [
    Line2D([0], [0], color="tab:blue", lw=2.5, label="Baseline – Avg"),
    Line2D([0], [0], color="tab:green", lw=2.5, label=r"Baseline – $\pi_{0.05}$"),
    Line2D([0], [0], color="tab:orange", lw=2.5, label=r"Baseline – $\pi_{0.01}$"),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="blue",
        markersize=11,
        markeredgecolor="black",
        label="Shiro – Avg",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="green",
        markersize=11,
        markeredgecolor="black",
        label=r"Shiro – $\pi_{0.05}$",
    ),
    Line2D(
        [0],
        [0],
        marker="*",
        color="w",
        markerfacecolor="orange",
        markersize=11,
        markeredgecolor="black",
        label=r"Shiro – $\pi_{0.01}$",
    ),
    Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        markerfacecolor="blue",
        markersize=10,
        markeredgecolor="black",
        label="Ada – Avg",
    ),
    Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        markerfacecolor="green",
        markersize=10,
        markeredgecolor="black",
        label=r"Ada – $\pi_{0.05}$",
    ),
    Line2D(
        [0],
        [0],
        marker="X",
        color="w",
        markerfacecolor="orange",
        markersize=10,
        markeredgecolor="black",
        label=r"Ada – $\pi_{0.01}$",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        markerfacecolor="grey",
        markersize=10,
        markeredgecolor="black",
        label="Progress",
    ),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=len(legend_handles),
    fontsize=9,
    framealpha=0.95,
    edgecolor="#cccccc",
    bbox_to_anchor=(0.5, 0.0),
)
import os

os.makedirs("research/img", exist_ok=True)
plt.savefig("research/img/visualization_final.png", dpi=300)
print("Saved to research/img/visualization_final.png")

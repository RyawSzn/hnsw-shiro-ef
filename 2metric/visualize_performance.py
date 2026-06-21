import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import seaborn as sns
from scipy import stats

DATASET    = "glove-100-angular"
BINS       = "20x20"
EXPERIMENTS = "/home/ryawszn/experiments"

LOOKUP_CSV  = f"{EXPERIMENTS}/2metric/lookup/lookup_table_{DATASET}_{BINS}.csv"
DIAG_CSV    = f"{EXPERIMENTS}/2metric/lookup/diagnostic_{DATASET}_{BINS}.csv"
CMP_CSV     = f"{EXPERIMENTS}/2metric/compare/comparison_{DATASET}.csv"
OUT_P1      = f"{EXPERIMENTS}/2metric/lookup/performance_{DATASET}_{BINS}.png"
OUT_P2      = f"{EXPERIMENTS}/2metric/lookup/performance_{DATASET}_{BINS}_compare.png"

RUNTIME = {
    "search_ms":    77582,
    "avg_recall":   0.99517,
    "avg_ef":       4869.03,
    "pct5_recall":  1.0,
    "pct1_recall":  0.9,
    "target_recall":0.95,
    "max_ef":       5000,
    "nq":           10000,
}

EF_DIST = {
    4800:7,4812:3,4819:17,4825:9,4829:1,4831:4,4834:12,4835:2,
    4836:27,4837:3,4838:6,4840:2,4841:4,4842:56,4843:3,4844:25,
    4845:39,4846:11,4847:118,4848:165,4849:65,4850:97,4851:154,
    4852:60,4853:142,4854:147,4855:114,4856:151,4857:280,4858:243,
    4859:186,4860:281,4861:442,4862:556,4863:446,4864:763,4865:1182,
    4866:1141,4867:932,4868:1304,4869:74,4900:281,4950:35,5000:410,
}

def extract_lower(s):
    try:
        return float(str(s).split(",")[0].replace("(","").strip())
    except Exception:
        return -1.0

def dark_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#cccccc")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444455")
    if xlabel: ax.set_xlabel(xlabel, fontsize=10, color="#cccccc")
    if ylabel: ax.set_ylabel(ylabel, fontsize=10, color="#cccccc")
    if title:  ax.set_title(title, fontsize=11, color="white", pad=7)
    ax.xaxis.label.set_color("#cccccc")
    ax.yaxis.label.set_color("#cccccc")

sns.set_theme(style="darkgrid", font_scale=1.05)

ldf = pd.read_csv(LOOKUP_CSV)
ldf["RC_sort"] = ldf["RC_bin"].apply(extract_lower)
ldf["RV_sort"] = ldf["RV_bin"].apply(extract_lower)
ldf = ldf.sort_values(["RC_sort","RV_sort"])
sorted_rc = sorted(ldf["RC_bin"].unique(), key=extract_lower)
sorted_rv = sorted(ldf["RV_bin"].unique(), key=extract_lower)
pivot_ef     = ldf.pivot(index="RC_bin",columns="RV_bin",values="ef").reindex(index=sorted_rc,columns=sorted_rv)
pivot_recall = ldf.pivot(index="RC_bin",columns="RV_bin",values="actual_recall").reindex(index=sorted_rc,columns=sorted_rv)

ddf = pd.read_csv(DIAG_CSV)
log_ef = np.log(ddf["ef_true"].replace(0,1))
rho_RC, _ = stats.spearmanr(ddf["RC"],   log_ef)
rho_RV, _ = stats.spearmanr(ddf["RV_rank"], log_ef)

cdf = pd.read_csv(CMP_CSV)
cdf["ratio"]  = cdf["2metric_ef"] / cdf["true_ef"].clip(lower=1)
cdf["error"]  = cdf["2metric_ef"] - cdf["true_ef"]
cdf["under"]  = cdf["2metric_ef"] < cdf["true_ef"]

ef_vals = np.array(list(EF_DIST.keys()))
ef_cnts = np.array(list(EF_DIST.values()))

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Lookup table + runtime overview
# ═══════════════════════════════════════════════════════════════════════════════
fig1 = plt.figure(figsize=(26,22), facecolor="#0f1117")
fig1.suptitle(
    f"2-Metric Adaptive EF — Performance Dashboard\n"
    f"{DATASET}  |  20×20 grid  |  target recall {RUNTIME['target_recall']:.2f}",
    fontsize=20, fontweight="bold", color="white", y=0.98,
)
gs1 = gridspec.GridSpec(3,3, figure=fig1, hspace=0.44, wspace=0.38)

ax_he  = fig1.add_subplot(gs1[0,:2])
ax_hr  = fig1.add_subplot(gs1[1,:2])
ax_sc  = fig1.add_subplot(gs1[0,2])
ax_efd = fig1.add_subplot(gs1[1,2])
ax_cdf = fig1.add_subplot(gs1[2,0])
ax_met = fig1.add_subplot(gs1[2,1])
ax_cor = fig1.add_subplot(gs1[2,2])

# heatmap – ef
sns.heatmap(pivot_ef, ax=ax_he, annot=True, fmt=".0f", cmap="YlOrRd",
            annot_kws={"size":7}, cbar_kws={"label":"Assigned ef","shrink":0.8},
            linewidths=0.3, linecolor="#333344")
dark_ax(ax_he, "Lookup Table — Assigned EF per (RC, RV) Bin",
        f"RV_Rank Interval  (ρ={rho_RV:+.3f} vs log ef_true)",
        f"RC Interval  (ρ={rho_RC:+.3f} vs log ef_true)")
ax_he.set_xticklabels(ax_he.get_xticklabels(), rotation=45, ha="right", fontsize=6)
ax_he.set_yticklabels(ax_he.get_yticklabels(), rotation=0, fontsize=6)
ax_he.collections[0].colorbar.ax.tick_params(colors="white")
ax_he.collections[0].colorbar.set_label("Assigned ef", color="white")

# heatmap – recall; highlight bins below target
min_r = ldf["actual_recall"].min() - 0.003
sns.heatmap(pivot_recall, ax=ax_hr, annot=True, fmt=".3f", cmap="crest",
            vmin=min_r, vmax=1.0,
            annot_kws={"size":7}, cbar_kws={"label":"Achieved Recall","shrink":0.8},
            linewidths=0.3, linecolor="#333344")
dark_ax(ax_hr, "Lookup Table — Actual Recall Achieved per Bin", "RV_Rank Interval", "RC Interval")
ax_hr.set_xticklabels(ax_hr.get_xticklabels(), rotation=45, ha="right", fontsize=6)
ax_hr.set_yticklabels(ax_hr.get_yticklabels(), rotation=0, fontsize=6)
for text in ax_hr.texts:
    try:
        val = float(text.get_text())
        if val < RUNTIME["target_recall"]:
            text.set_color("#ff4444")
            text.set_fontweight("bold")
    except ValueError:
        pass
ax_hr.collections[0].colorbar.ax.tick_params(colors="white")
ax_hr.collections[0].colorbar.set_label("Achieved Recall", color="white")

# scatter: RC vs RV coloured by log(ef_true)
sc = ax_sc.scatter(ddf["RV_rank"], ddf["RC"],
                   c=np.log(ddf["ef_true"].clip(lower=1)),
                   cmap="plasma", s=12, alpha=0.65, rasterized=True)
cb = plt.colorbar(sc, ax=ax_sc)
cb.set_label("log(ef_true)", color="white")
cb.ax.tick_params(colors="white")
dark_ax(ax_sc, "Query Hardness Space\n(color = log ef_true)", "RV_Rank (entrapment)", "RC (relative contrast)")

# ef distribution histogram
ax_efd.bar(ef_vals, ef_cnts, width=1.8, color="#5c9cf5", alpha=0.85, edgecolor="none")
ax_efd.axvline(RUNTIME["avg_ef"], color="#ff6b6b", linewidth=2, linestyle="--",
               label=f"avg ef = {RUNTIME['avg_ef']:.0f}")
dark_ax(ax_efd, "EF Distribution (runtime)", "Assigned EF (with avg-ef floor)", "# Queries")
ax_efd.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")
ax_efd.tick_params(axis="x", rotation=45)

# CDF of ef_true
sorted_ef_true = np.sort(ddf["ef_true"])
cdf_y = np.arange(1, len(sorted_ef_true)+1) / len(sorted_ef_true)
ax_cdf.plot(sorted_ef_true, cdf_y, color="#a8e6cf", linewidth=2)
ax_cdf.axvline(ddf["ef_true"].mean(), color="#ff8b94", linewidth=1.5, linestyle="--",
               label=f"mean ef_true = {ddf['ef_true'].mean():.0f}")
ax_cdf.axvline(RUNTIME["max_ef"], color="#ffd166", linewidth=1, linestyle=":",
               label=f"max_ef = {RUNTIME['max_ef']}")
dark_ax(ax_cdf, "CDF of Per-Query ef_true\n(calibration sample)", "ef_true (min ef to hit target recall)", "CDF")
ax_cdf.set_ylim(0, 1.05)
ax_cdf.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")

# metrics table
ax_met.axis("off")
ax_met.set_facecolor("#12151f")
for sp in ax_met.spines.values(): sp.set_visible(False)
rows = [
    ("Avg Recall",      f"{RUNTIME['avg_recall']:.4f}",  "#a8e6cf"),
    ("Avg EF Used",     f"{RUNTIME['avg_ef']:.1f}",      "#5c9cf5"),
    ("5th %ile Recall", f"{RUNTIME['pct5_recall']:.3f}", "#a8e6cf"),
    ("1st %ile Recall", f"{RUNTIME['pct1_recall']:.3f}", "#ffd3b6"),
    ("Search Time",     f"{RUNTIME['search_ms']/1000:.1f}s", "#d4a5ff"),
    ("# Queries",       f"{RUNTIME['nq']:,}",            "#cccccc"),
    ("Max EF",          f"{RUNTIME['max_ef']:,}",        "#cccccc"),
    ("Target Recall",   f"{RUNTIME['target_recall']:.2f}","#ffd3b6"),
]
for i,(label,value,color) in enumerate(rows):
    y = 0.93 - i*0.115
    ax_met.text(0.05, y, label, transform=ax_met.transAxes, fontsize=12, color="#aaaaaa", va="top")
    ax_met.text(0.95, y, value, transform=ax_met.transAxes, fontsize=13, fontweight="bold",
                color=color, va="top", ha="right")
ax_met.set_title("Runtime Metrics", fontsize=12, color="white", pad=6)

# correlation panel
log_ef_d = np.log(ddf["ef_true"].clip(lower=1))
ax_cor.scatter(ddf["RV_rank"], log_ef_d, c="#5c9cf5", s=8, alpha=0.4, rasterized=True, label="data (RV)")
x_rv = np.linspace(ddf["RV_rank"].min(), ddf["RV_rank"].max(), 200)
m,b = np.polyfit(ddf["RV_rank"], log_ef_d, 1)
ax_cor.plot(x_rv, m*x_rv+b, color="#ff6b6b", lw=2, label=f"RV fit  ρ={rho_RV:+.3f}")
ax_twin = ax_cor.twiny()
x_rc = np.linspace(ddf["RC"].min(), ddf["RC"].max(), 200)
m2,b2 = np.polyfit(ddf["RC"], log_ef_d, 1)
ax_twin.plot(x_rc, m2*x_rc+b2, color="#ffd166", lw=2, linestyle="--", label=f"RC fit  ρ={rho_RC:+.3f}")
ax_twin.set_xlabel("RC", fontsize=9, color="#ffd166")
ax_twin.tick_params(colors="#ffd166")
ax_twin.spines["top"].set_edgecolor("#ffd166")
dark_ax(ax_cor, "Metric Correlation with\nlog(ef_true)", "RV_Rank", "log(ef_true)")
l1,lb1 = ax_cor.get_legend_handles_labels()
l2,lb2 = ax_twin.get_legend_handles_labels()
ax_cor.legend(l1+l2, lb1+lb2, fontsize=8, facecolor="#222233", labelcolor="white", edgecolor="#555566", loc="upper right")
ax_cor.tick_params(colors="#5c9cf5")
ax_cor.spines["bottom"].set_edgecolor("#5c9cf5")

os.makedirs(os.path.dirname(OUT_P1), exist_ok=True)
fig1.savefig(OUT_P1, dpi=180, bbox_inches="tight", facecolor=fig1.get_facecolor())
print(f"Page 1 saved: {OUT_P1}")
plt.close(fig1)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prediction accuracy vs ground-truth ef
# ═══════════════════════════════════════════════════════════════════════════════
fig2 = plt.figure(figsize=(24,18), facecolor="#0f1117")
fig2.suptitle(
    f"2-Metric Prediction Accuracy vs Ground-Truth EF\n"
    f"{DATASET}  |  n={len(cdf)} queries  |  target recall {RUNTIME['target_recall']:.2f}",
    fontsize=18, fontweight="bold", color="white", y=0.98,
)
gs2 = gridspec.GridSpec(2,3, figure=fig2, hspace=0.42, wspace=0.36)

ax_pv  = fig2.add_subplot(gs2[0,0])
ax_err = fig2.add_subplot(gs2[0,1])
ax_rat = fig2.add_subplot(gs2[0,2])
ax_rv  = fig2.add_subplot(gs2[1,0])
ax_rc  = fig2.add_subplot(gs2[1,1])
ax_sum = fig2.add_subplot(gs2[1,2])

# predicted vs true scatter
under_mask = cdf["under"]
ax_pv.scatter(cdf.loc[~under_mask,"true_ef"], cdf.loc[~under_mask,"2metric_ef"],
              c="#5c9cf5", s=14, alpha=0.55, rasterized=True, label="over-allocated")
ax_pv.scatter(cdf.loc[under_mask,"true_ef"],  cdf.loc[under_mask,"2metric_ef"],
              c="#ff6b6b", s=18, alpha=0.85, rasterized=True, label=f"under-predicted ({under_mask.sum()})")
mx = max(cdf["true_ef"].max(), cdf["2metric_ef"].max())
ax_pv.plot([0,mx],[0,mx], color="#ffd166", lw=1.5, linestyle="--", label="perfect prediction")
dark_ax(ax_pv, "2-Metric Predicted EF vs True EF", "true_ef (ground truth)", "2metric_ef (predicted)")
ax_pv.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")

# error histogram
bins_e = np.linspace(cdf["error"].min(), cdf["error"].max(), 60)
ax_err.hist(cdf["error"], bins=bins_e, color="#5c9cf5", alpha=0.8, edgecolor="none")
ax_err.axvline(0, color="#ffd166", lw=1.5, linestyle="--")
ax_err.axvline(cdf["error"].mean(), color="#ff8b94", lw=2, linestyle="-",
               label=f"mean error = {cdf['error'].mean():+.0f}")
ax_err.axvline(np.median(cdf["error"]), color="#a8e6cf", lw=2, linestyle="-.",
               label=f"median error = {np.median(cdf['error']):+.0f}")
dark_ax(ax_err, "Prediction Error Distribution\n(2metric_ef − true_ef)", "Error (ef units)", "# Queries")
ax_err.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")

# ratio histogram (log scale x)
ratio_clip = cdf["ratio"].clip(upper=20)
ax_rat.hist(ratio_clip, bins=40, color="#a8e6cf", alpha=0.8, edgecolor="none")
ax_rat.axvline(1.0, color="#ff6b6b", lw=2, linestyle="--", label="ratio=1 (perfect)")
ax_rat.axvline(cdf["ratio"].median(), color="#ffd166", lw=2, linestyle="-.",
               label=f"median ratio = {cdf['ratio'].median():.1f}×")
dark_ax(ax_rat, "Over-allocation Ratio\n(2metric_ef / true_ef, clipped @20)", "2metric_ef / true_ef", "# Queries")
ax_rat.legend(fontsize=9, facecolor="#222233", labelcolor="white", edgecolor="#555566")

# RV_rank vs error scatter
sc2 = ax_rv.scatter(cdf["RV_rank"], cdf["error"],
                    c=np.log(cdf["true_ef"].clip(lower=1)),
                    cmap="plasma", s=12, alpha=0.6, rasterized=True)
ax_rv.axhline(0, color="#ffd166", lw=1, linestyle="--")
cb2 = plt.colorbar(sc2, ax=ax_rv)
cb2.set_label("log(true_ef)", color="white")
cb2.ax.tick_params(colors="white")
dark_ax(ax_rv, "Prediction Error vs RV_Rank\n(color = log true_ef)", "RV_Rank", "error (2metric − true)")

# RC vs error scatter
sc3 = ax_rc.scatter(cdf["RC"], cdf["error"],
                    c=np.log(cdf["true_ef"].clip(lower=1)),
                    cmap="plasma", s=12, alpha=0.6, rasterized=True)
ax_rc.axhline(0, color="#ffd166", lw=1, linestyle="--")
cb3 = plt.colorbar(sc3, ax=ax_rc)
cb3.set_label("log(true_ef)", color="white")
cb3.ax.tick_params(colors="white")
dark_ax(ax_rc, "Prediction Error vs RC\n(color = log true_ef)", "RC (relative contrast)", "error (2metric − true)")

# summary stats panel
ax_sum.axis("off")
ax_sum.set_facecolor("#12151f")
for sp in ax_sum.spines.values(): sp.set_visible(False)
n_under = under_mask.sum()
n_over  = (~under_mask & (cdf["2metric_ef"] != cdf["true_ef"])).sum()
n_exact = (cdf["2metric_ef"] == cdf["true_ef"]).sum()
rows2 = [
    ("Sample Size",        f"{len(cdf)}",                         "#cccccc"),
    ("MAE",                f"{np.abs(cdf['error']).mean():.1f}",  "#ffd3b6"),
    ("Median Error",       f"{np.median(cdf['error']):+.0f}",     "#ffd3b6"),
    ("Under-predictions",  f"{n_under}  ({100*n_under/len(cdf):.1f}%)", "#ff6b6b"),
    ("Over-predictions",   f"{n_over}  ({100*n_over/len(cdf):.1f}%)",   "#5c9cf5"),
    ("Exact hits",         f"{n_exact}  ({100*n_exact/len(cdf):.1f}%)", "#a8e6cf"),
    ("Median ratio",       f"{cdf['ratio'].median():.1f}×",       "#ffd166"),
    ("95th %ile ratio",    f"{np.percentile(cdf['ratio'],95):.1f}×","#ffd166"),
    ("Under-est. risk",    f"{100*n_under/len(cdf):.1f}%",        "#ff6b6b"),
]
for i,(label,value,color) in enumerate(rows2):
    y = 0.96 - i*0.105
    ax_sum.text(0.05, y, label, transform=ax_sum.transAxes, fontsize=12, color="#aaaaaa", va="top")
    ax_sum.text(0.95, y, value, transform=ax_sum.transAxes, fontsize=12, fontweight="bold",
                color=color, va="top", ha="right")
ax_sum.set_title("Prediction Accuracy Summary", fontsize=12, color="white", pad=6)

fig2.savefig(OUT_P2, dpi=180, bbox_inches="tight", facecolor=fig2.get_facecolor())
print(f"Page 2 saved: {OUT_P2}")
plt.close(fig2)

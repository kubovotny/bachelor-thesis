"""
Figure 3.6c (revised) – Correlation Heatmap + Lag Profile

Two panels:
  A) Heatmap: top features × eras (same as before but with v2 data)
  B) Lag profile: how r(finbert_mean, MRO_diff) evolves with lag 0–15
     Key finding: r peaks at lag ≈ 11 meetings (~1.4 years ahead)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/correlations"

ERA_ORDER  = ["Pre-ZLB", "ZLB", "Hiking", "Full"]
ERA_LABELS = ["Pre-ZLB\n(1999–2011)", "ZLB\n(2012–2021)",
              "Hiking\n(2022–2024)", "Full\n(1999–2024)"]


# ── Load CSVs ─────────────────────────────────────────────────────────────────
def load_data():
    piv  = pd.read_csv(OUTPUT_DIR / "table_summary.csv")
    corr = pd.read_csv(OUTPUT_DIR / "table_feature.csv")
    corr["sig"] = corr["sig"].fillna("")
    corr = corr.drop_duplicates(subset=["Feature", "Era"])
    return piv, corr


# ── Pretty feature labels ─────────────────────────────────────────────────────
def prettify(name: str) -> str:
    lag = ""
    if ":lag" in name:
        lag_n = name.split(":lag")[1]
        name  = name.split(":lag")[0]
        lag   = f" [t−{lag_n}]"
    name = (name.replace("finbert", "FB").replace("roberta", "RB")
                .replace("_mean", " μ").replace("_std", " σ")
                .replace("_max", " max").replace("_min", " min")
                .replace("_IS_", " IS·").replace("_QA_", " QA·")
                .replace("_MP_", " MP·").replace("_EP_", " EP·")
                .replace("_FS_", " FS·").replace("_OI_", " OI·")
                .replace("_IS", " IS").replace("_QA", " QA"))
    return name.strip() + lag


# ── Figure ────────────────────────────────────────────────────────────────────
def plot_fig_3_6c(top_n: int = 18, save: bool = True):
    piv, corr = load_data()
    piv = piv.head(top_n)
    piv["Label"] = piv["Feature"].apply(prettify)

    # ── GRAF A: Heatmap (Sila prepojenia) ──────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    cols = [c for c in ERA_ORDER if c in piv.columns]
    
    # Oprava imshow: vynútime float a ošetríme NaN
    val_mat = piv[cols].astype(float).fillna(0).values 
    
    cmap = plt.cm.RdYlGn
    vmax = 0.55
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    
    # Tu sa deje kúzla – ak imshow "nič nerobil", teraz bude
    im = ax1.imshow(val_mat, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")

    # Lepší nadpis (Title)
    ax1.set_title("The Echo of Central Bank Words: How Sentiment Shapes Interest Rates", 
                  fontsize=14, pad=20)
    
    # Tvoja logika popisov zostáva
    ax1.set_xticks(range(len(cols)))
    ax1.set_xticklabels([ERA_LABELS[ERA_ORDER.index(c)] for c in cols], fontsize=9)
    ax1.set_yticks(range(len(piv)))
    ax1.set_yticklabels(piv["Label"].values, fontsize=8, fontfamily="monospace")
    ax1.xaxis.set_ticks_position("top")

    # Pridanie textu do buniek
    for i in range(len(piv)):
        for j in range(len(cols)):
            val = val_mat[i, j]
            ax1.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, 
                     color="white" if abs(val) > 0.3 else "black")

    if save: fig1.savefig(OUTPUT_DIR / "fig_heatmap.pdf", bbox_inches="tight")


    # ── GRAF 2: Lag profile (Panel B - pôvodná orientácia) ──────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    lag_features = [
        ("finbert_mean",    "FinBERT overall",    "#2c6fad", "Full"),
        ("finbert_IS_mean", "FinBERT IS",         "#27ae60", "Full"),
        ("roberta_IS_mean", "RoBERTa IS",         "#c0392b", "Full"),
    ]

    for base_feat, label, color, era in lag_features:
        lag_data = corr[(corr["Base"] == base_feat) & (corr["Era"] == era)].sort_values("Lag")
        if lag_data.empty: continue

        # Vrátené späť: x=Lag, y=r
        ax2.plot(lag_data["Lag"], lag_data["r"],
                 marker="o", markersize=5, linewidth=1.8, color=color, label=label)

        # Mark peak
        peak = lag_data.loc[lag_data["r"].abs().idxmax()]
        ax2.scatter([peak["Lag"]], [peak["r"]], s=80, color=color, marker="*", zorder=5)
        ax2.text(peak["Lag"] - 0.8, peak["r"] + 0.03,
                 f"  lag={int(peak['Lag'])}\n  r={peak['r']:.3f}",
                 fontsize=7.5, color=color, va="center")

    ax2.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax2.axvline(0, color="#ddd", linewidth=0.6)
    ax2.set_xlabel("Lag (Number of meetings)")
    ax2.set_ylabel("Correlation Strength (r)")
    ax2.set_title("Time Sensitivity: When does the market react?", fontsize=10, pad=6)
    ax2.legend(fontsize=8.5, framealpha=0.92, loc="upper right")
    ax2.grid(alpha=0.22, linewidth=0.6)
    ax2.set_xlim(-0.5, 15.5)
    ax2.set_ylim(-0.12,0.65)

    if save:
        fig2.savefig(OUTPUT_DIR / "fig_lag_profile.pdf", bbox_inches="tight")

    return fig1, fig2
if __name__ == "__main__":
    plot_fig_3_6c(top_n=10)
    plt.show()
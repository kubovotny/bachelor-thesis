"""
Section 3.5.2 – Predicting the Direction of ECB Rate Changes
(v3 – fixes PerformanceWarning; redesigns figs to 3 publication plots)

Changes from v2:
  - Lag columns built with pd.concat (eliminates DataFrame fragmentation warning)
  - Three figures instead of four:
      fig_performance.pdf        : all-model performance bar chart
      fig_comparison_m3_m4.pdf   : M3 vs M4 confusion matrices side by side
      fig_feature_analysis.pdf   : M3 coefficients (left) + M4 importance (right)
  - fig_rf_importance.pdf dropped (merged into fig_feature_analysis)

Four models
  M1 Baseline        : [mro_lag, mro_change_lag]
  M2 Sentiment       : [finbert_IS_mean_lag2, finbert_mean_lag5,
                        finbert_mean_lag2,    finbert_IS_std]
  M3 Augmented Logit : M1 ∪ M2 features
  M4 Random Forest   : same feature set as M3, non-linear estimator

Evaluation: TimeSeriesSplit (n=5) — accuracy ± std + AUC OvR
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                              classification_report, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORD_LIMIT   = 1
N_SPLITS     = 5
RANDOM_STATE = 42

SENT_FEATS = [
    "finbert_mean:lag4",
    "finbert_QA_std",
    "finbert_IS_min:lag2",
    "finbert_QA_max:lag2"
]


# ── pretty labels for axes (no LaTeX escaping – plain matplotlib rendering) ──
FEAT_LABELS = {
    "mro_lag":              "MRO [t−1]",
    "mro_change_lag":       "ΔMRO [t−1]",
    "finbert_mean_lag4":    "FB overall μ [t−4]",
    "finbert_QA_std":    "FB overall μ",
    "finbert_IS_min_lag2": "FB IS MP μ [t−2]",
    "finbert_QA_max:lag2":       "FB QA max [t-2]",
}


# ── Data loading (PerformanceWarning fix) ─────────────────────────────────────
def _load_raw() -> pd.DataFrame:
    """Merge the two model-data configurations into one wide DataFrame."""
    configs = [
        dict(IS_QA_division=False, qa_options="both_together", with_label=False),
        dict(IS_QA_division=True,  qa_options="just_answers",  with_label=False),
    ]
    base = None
    for cfg in configs:
        d = return_data(
            market_data="ECB Money Market.xlsx",
            word_limit=WORD_LIMIT,
            **cfg,
        )
        if base is None:
            base = d
        else:
            new_cols = [c for c in d.columns if c not in base.columns and c != "date"]
            base = pd.merge(base, d[["date"] + new_cols], on="date", how="left")
    return base.sort_values("date").reset_index(drop=True)


def load_data() -> pd.DataFrame:
    df = _load_raw()
    df["date"]     = pd.to_datetime(df["date"])
    df["MRO_diff"] = df["MRO"].diff()
    df["direction"] = np.sign(df["MRO_diff"]).fillna(0).astype(int)

    # ── Build ALL lag columns in one pd.concat call (no fragmentation) ────────
    sent_cols = [
        c for c in df.columns
        if any(m in c for m in ["finbert", "roberta"])
        and c.endswith(("_mean", "_std", "_max", "_min"))
    ]
    lag_frames = [df]
    for col in sent_cols:
        for lag in [1, 2, 3, 5, 7, 8, 11, 13]:
            lag_frames.append(
                df[col].shift(lag).rename(f"{col}_lag{lag}").to_frame()
            )
    lag_frames.append(df["MRO_diff"].shift(1).rename("mro_change_lag").to_frame())
    lag_frames.append(df["MRO"].shift(1).rename("mro_lag").to_frame())

    return pd.concat(lag_frames, axis=1).copy()   # .copy() defragments


# ── Helpers ───────────────────────────────────────────────────────────────────
def _Xy(df: pd.DataFrame, feats: list[str]):
    avail = [f for f in feats if f in df.columns]
    clean = df[avail + ["direction"]].dropna()
    return clean[avail], clean["direction"]


def _scale(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def _cv(X: pd.DataFrame, y: pd.Series, model, scale=True, n=N_SPLITS):
    tscv = TimeSeriesSplit(n_splits=n)
    accs, aucs = [], []
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        if scale:
            Xtr, Xte = _scale(Xtr, Xte)
        model.fit(Xtr, ytr)
        accs.append((model.predict(Xte) == yte).mean())
        if len(np.unique(yte)) == 3:
            try:
                prob = model.predict_proba(Xte)
                aucs.append(roc_auc_score(yte, prob, multi_class="ovr",
                                          labels=[-1, 0, 1]))
            except Exception:
                pass
    return dict(acc_mean=np.mean(accs), acc_std=np.std(accs),
                auc_mean=np.mean(aucs) if aucs else np.nan)

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

def tune_logit(X: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    """Inner CV: tune C over a small grid."""
    inner_cv = TimeSeriesSplit(n_splits=3)
    grid = GridSearchCV(
        LogisticRegression(
            class_weight="balanced", solver="lbfgs",
            max_iter=2000, random_state=RANDOM_STATE
        ),
        param_grid={"C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]},
        cv=inner_cv,
        scoring="roc_auc_ovr",
        refit=True,
    )
    grid.fit(StandardScaler().fit_transform(X), y)
    print(f"  Best C={grid.best_params_['C']:.3f}  "
          f"inner AUC={grid.best_score_:.4f}")
    return grid.best_estimator_


# ── Model runs ────────────────────────────────────────────────────────────────
def run_models(df: pd.DataFrame) -> dict:
    sent = [f for f in SENT_FEATS if f in df.columns]
    mro  = ["mro_lag", "mro_change_lag"]

    logit = LogisticRegression(
        class_weight="balanced", solver="lbfgs",
        max_iter=2000, C=1.0, random_state=RANDOM_STATE,
    )
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=4, min_samples_leaf=3,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
    )

    specs = {
        "M1 Baseline":        (mro,        logit, True),
        "M2 Sentiment":       (sent,       logit, True),
        "M3 Augmented":       (mro + sent, logit, True),
        "M4 Random Forest":   (mro + sent, rf,    False),
    }

    results = {}
    for name, (feats, model, scale) in specs.items():
        X, y = _Xy(df, feats)
        cv   = _cv(X, y, model, scale=scale)
        print(f"\n--- {name} (n={len(X)}, feats={len(X.columns)}) ---")
        print(f"  CV Accuracy : {cv['acc_mean']:.4f} ± {cv['acc_std']:.4f}")
        if not np.isnan(cv["auc_mean"]):
            print(f"  CV AUC (OvR): {cv['auc_mean']:.4f}")

        # Full-sample fit for confusion matrix / coefficients
        Xf = X.values if not scale else StandardScaler().fit_transform(X)
        model.fit(Xf, y)

        results[name] = dict(cv=cv, model=model,
                             X=X, y=y, pred=model.predict(Xf),
                             feats=list(X.columns), classes=list(model.classes_))

    _, y_all = _Xy(df, ["mro_lag"])
    results["_naive"] = float((y_all == 0).mean())
    print(f"\nNaive baseline (always Hold): {results['_naive']:.4f}")
    return results


# ── Figure A – performance (all four models) ─────────────────────────────────
def fig_performance(results: dict, save=True):
    names  = ["M1 Baseline", "M2 Sentiment", "M3 Augmented", "M4 Random Forest"]
    colors = ["#95a5a6", "#c0392b", "#2c6fad", "#27ae60"]

    accs = [results[m]["cv"]["acc_mean"] for m in names]
    errs = [results[m]["cv"]["acc_std"]  for m in names]
    aucs = [results[m]["cv"]["auc_mean"] for m in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))

    ax.bar(x - 0.20, accs, 0.35, color=colors, alpha=0.85,
           yerr=errs, capsize=5, label="Accuracy")
    ax.bar(x + 0.20, aucs, 0.35, color=colors, alpha=0.38,
           edgecolor=colors, linewidth=1.5, label="AUC (one-vs-rest)")
    ax.axhline(results["_naive"], ls="--", lw=1.5, color="#888",
               label=f"Naive baseline (always Hold): {results['_naive']:.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score (0 to 1)")
    ax.set_title("Cross-validated performance of the four predictive models")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    if save:
        fig.savefig(OUTPUT_DIR / "fig_performance.pdf", bbox_inches="tight")
    return fig


# ── Figure B – M3 vs M4 confusion matrices ───────────────────────────────────
def fig_comparison_m3_m4(results: dict, save=True):
    """
    Side-by-side confusion matrices so the reader can directly compare
    the two best models without the baseline clutter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cls_labels = ["Cut", "Hold", "Hike"]

    for ax, name, title in zip(
        axes,
        ["M3 Augmented", "M4 Random Forest"],
        ["M3 — Augmented Logit", "M4 — Random Forest"],
    ):
        r  = results[name]
        cm = confusion_matrix(r["y"], r["pred"], labels=[-1, 0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=cls_labels)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title, fontsize=12, fontweight="bold")

        # annotate class-level recall in the title area
        recall = cm.diagonal() / cm.sum(axis=1)
        ax.set_xlabel(
            f"Predicted label\n"
            f"Recall — Cut: {recall[0]:.0%}  Hold: {recall[1]:.0%}  Hike: {recall[2]:.0%}",
            fontsize=10,
        )

    fig.suptitle(
        "Confusion matrices: M3 Augmented Logit vs M4 Random Forest",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig_comparison_m3_m4.pdf", bbox_inches="tight")
    return fig


# ── Figure C – M3 coefficients vs M4 feature importance ─────────────────────
def fig_feature_analysis(results: dict, save=True):
    """
    Left panel : M3 standardised coefficients by class (Cut / Hold / Hike)
    Right panel: M4 Random Forest feature importances, same feature ordering
    Both panels share the same y-axis feature list so the eye can compare
    what each model relies on.
    """
    r3 = results["M3 Augmented"]
    r4 = results["M4 Random Forest"]

    feats  = r3["feats"]            # same feature set for both
    labels = [FEAT_LABELS.get(f, f) for f in feats]
    n_feat = len(feats)

    # M3 coefficient matrix  (classes × features)
    coef_df = pd.DataFrame(
        r3["model"].coef_,
        columns=feats,
        index=[{-1: "Cut", 0: "Hold", 1: "Hike"}[c] for c in r3["classes"]],
    )

    # M4 feature importances, same feature order as M3
    imp_map  = dict(zip(r4["feats"], r4["model"].feature_importances_))
    imps     = np.array([imp_map.get(f, 0.0) for f in feats])

    # ── layout ────────────────────────────────────────────────────────────────
    fig1, ax_l = plt.subplots(figsize=(8, 5.5))

    # Left: M3 grouped bar (Cut / Hold / Hike)
    x     = np.arange(n_feat)
    w     = 0.25
    cls_c = {"Cut": "#c0392b", "Hold": "#95a5a6", "Hike": "#2c6fad"}
    for i, (cls, col) in enumerate(cls_c.items()):
        ax_l.bar(x + (i - 1) * w, coef_df.loc[cls], w,
                 label=cls, color=col, alpha=0.80)
    ax_l.axhline(0, color="#444", lw=0.8)
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
    ax_l.set_ylabel("Standardised coefficient")
    ax_l.set_title("M3 Augmented Logit\nstandardised coefficients by class",
                   fontsize=12)
    ax_l.legend(fontsize=10)
    ax_l.grid(axis="y", alpha=0.22)

    # Right: M4 horizontal bar, same feature order
    fig2, ax_r = plt.subplots(figsize=(8, 5.5))
    y_pos = np.arange(n_feat)
    ax_r.barh(y_pos, imps, color="#27ae60", alpha=0.85, edgecolor="white")
    ax_r.set_yticks(y_pos)
    ax_r.set_yticklabels(labels, fontsize=10)
    ax_r.set_xlabel("Feature importance (impurity decrease)")
    ax_r.set_title("M4 Random Forest\nfeature importance", fontsize=12)
    ax_r.grid(axis="x", alpha=0.22)

    fig1.tight_layout()
    fig2.tight_layout()
    if save:
        fig1.savefig(OUTPUT_DIR / "fig_coefficients.pdf", bbox_inches="tight")
        fig2.savefig(OUTPUT_DIR / "fig_rf_importancies.pdf", bbox_inches="tight")

    return fig1, fig2


# ── Classification report ─────────────────────────────────────────────────────
def report(results: dict):
    for name in ["M3 Augmented", "M4 Random Forest"]:
        r = results[name]
        print(f"\n{'='*60}")
        print(f"{name} — full-sample classification report")
        print("="*60)
        print(classification_report(
            r["y"], r["pred"],
            labels=[-1, 0, 1],
            target_names=["Cut", "Hold", "Hike"],
        ))


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    d  = df["direction"]
    print(f"  {len(df)} meetings | "
          f"Hike: {(d==1).sum()} | Cut: {(d==-1).sum()} | Hold: {(d==0).sum()}")

    results = run_models(df)
    fig_performance(results)
    fig_comparison_m3_m4(results)
    fig_feature_analysis(results)
    report(results)
    plt.show()
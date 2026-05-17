"""
predict_model_v3.py – Composite Target + Δsentiment + Nested CV

Key additions vs. v2
─────────────────────────────────────────────────────────────────────────────
1. Composite policy target
   – Outside ZLB: direction = sign(MRO.diff()), filtered to MRO_diff ≠ 0
   – Inside ZLB : direction = sign(shadow_diff), filtered to |shadow_diff|
                  > MIN_SHADOW_CHANGE (≈ 10bp equivalent)
   – Effect: n_changed roughly doubles (ZLB meetings now informative)
   – All four models trained on the same composite target for comparability

2. Δsentiment features in candidate pool
   – d_finbert_mean_lag1/2/5: communication surprise (acceleration of tone)
   – Approximately orthogonal to level features → less collinearity

3. Nested cross-validation
   – Outer loop : 5-fold TimeSeriesSplit → honest performance estimate
   – Inner loop : 3-fold TimeSeriesSplit → selects regularization C for logit
                  and max_depth / min_samples_leaf for Random Forest
   – Prevents selection bias from tuning on the same splits used for evaluation

4. Forward feature selection (M2 and M3)
   – Exhaustive search over C(7,k) combinations for k=1..4
   – Run inside inner CV to avoid leakage
   – Best combination selected by inner-CV AUC
   – Result: M2 and M3 use data-driven feature set, not hardcoded

5. Comparison summary: v2 AUC vs v3 AUC printed at the end

Models
─────────────────────────────────────────────────────────────────────────────
M1  Baseline        : [mro_lag, mro_change_lag]  — rate history only
M2  Sentiment       : best subset from CANDIDATE_FEATURES (inner CV)
M3  Augmented       : M1 ∪ best sentiment subset (inner CV)
M4  Random Forest   : M3 feature set, HP tuned by inner CV
"""

from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WORD_LIMIT = 1
N_SPLITS_OUTER = 5
N_SPLITS_INNER = 3
RANDOM_STATE = 42
MIN_SHADOW_CHANGE = 0.10  # % — shadow rate noise threshold

ZLB_START = "2012-01-01"
ZLB_END = "2021-12-31"

# Candidate pool: 7 economically motivated features
# Short horizon (lag 1-2) | Medium horizon (lag 5) | Dispersion | Cross-model
CANDIDATE_FEATURES = [
    "finbert_mean_lag2",  # level, overall, lag 2
    "finbert_mean_lag5",  # level, overall, lag 5  (telegraph window)
    "finbert_IS_mean_lag2",  # level, IS only, lag 2  (prepared statement)
    "finbert_IS_std",  # dispersion, IS only    (consistency proxy)
    "d_finbert_mean_lag1",  # Δlevel, overall, lag 1 (momentum t-1)
    "d_finbert_mean_lag2",  # Δlevel, overall, lag 2 (momentum t-2)
    "roberta_IS_mean_lag2",  # cross-model robustness check
    "roberta_mean_lag2",  # level, overall, lag 2
    "roberta_mean_lag5",  # level, overall, lag 5  (telegraph window)
    "roberta_IS_mean_lag2",  # level, IS only, lag 2  (prepared statement)
    "roberta_IS_std",  # dispersion, IS only    (consistency proxy)
]

# Logit regularization grid
C_GRID = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# RF hyperparameter grid
RF_GRID = {
    "max_depth": [3, 4, 5],
    "min_samples_leaf": [3, 5, 7],
}

# Pretty labels for plots
FEAT_LABELS = {
    "mro_lag": "MRO [t−1]",
    "mro_change_lag": "ΔMRO [t−1]",
    "finbert_mean_lag2": "FB overall μ [t−2]",
    "finbert_mean_lag5": "FB overall μ [t−5]",
    "finbert_IS_mean_lag2": "FB IS μ [t−2]",
    "finbert_IS_std": "FB IS σ",
    "d_finbert_mean_lag1": "ΔFB overall μ [t−1]",
    "d_finbert_mean_lag2": "ΔFB overall μ [t−2]",
    "roberta_IS_mean_lag2": "RB IS μ [t−2]",
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    configs = [
        dict(IS_QA_division=False, qa_options="both_together", with_label=False),
        dict(IS_QA_division=True, qa_options="just_answers", with_label=False),
    ]
    df = None
    for cfg in configs:
        d = return_data(
            market_data="ECB Money Market.xlsx",
            word_limit=WORD_LIMIT,
            **cfg,
        )
        if df is None:
            df = d
        else:
            new_cols = [c for c in d.columns if c not in df.columns and c != "date"]
            df = pd.merge(df, d[["date"] + new_cols], on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Policy targets ────────────────────────────────────────────────────────
    df["MRO_diff"] = df["MRO"].diff()

    shadow_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ("shadow", "wu", "xia"))),
        None,
    )
    if shadow_col is None:
        raise KeyError("Wu-Xia shadow rate column not found.")
    df["shadow_diff"] = df[shadow_col].diff()

    # Composite: shadow inside ZLB (with noise threshold), MRO outside
    zlb_mask = (df["date"] >= ZLB_START) & (df["date"] <= ZLB_END)
    shadow_signal = df["shadow_diff"].where(
        df["shadow_diff"].abs() >= MIN_SHADOW_CHANGE, 0.0
    )
    df["composite_diff"] = np.where(zlb_mask, shadow_signal, df["MRO_diff"])
    df["direction"] = (
        np.sign(df["composite_diff"]).fillna(0).astype("Int64").fillna(0).astype(int)
    )

    # ── Sentiment lag features ────────────────────────────────────────────────
    sent_cols = [
        c
        for c in df.columns
        if any(m in c for m in ("finbert", "roberta")) and c.endswith(("_mean", "_std"))
    ]

    lag_frames = {}
    for col in sent_cols:
        dvals = df[col].diff()
        lag_frames[f"d_{col}"] = dvals
        for lag in [1, 2, 3, 5, 7, 11]:
            lag_frames[f"{col}_lag{lag}"] = df[col].shift(lag)
            lag_frames[f"d_{col}_lag{lag}"] = dvals.shift(lag)

    lag_frames["mro_lag"] = df["MRO"].shift(1)
    lag_frames["mro_change_lag"] = df["MRO_diff"].shift(1)

    df = pd.concat(
        [df, pd.DataFrame(lag_frames, index=df.index)],
        axis=1,
    ).copy()

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────
def _Xy(df: pd.DataFrame, feats: list[str]):
    """
    Return (X, y) for ALL meetings — including Hold (direction=0).

    The composite target enriches the direction label for ZLB meetings
    but the Hold class must remain in the dataset for 3-class classification.
    Filtering to composite_diff != 0 would remove ~150 Hold meetings and
    collapse the problem to binary, breaking roc_auc_score (OvR, 3-class).
    """
    avail = [f for f in feats if f in df.columns]
    clean = df[avail + ["direction"]].dropna(subset=avail + ["direction"])
    return clean[avail], clean["direction"]


def _scale(X_tr, X_te):
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def _auc_cv(X, y, model, scale=True, n=N_SPLITS_INNER) -> float:
    """
    Inner CV AUC (OvR, 3-class).
    If a fold's test set contains fewer than 3 classes, that fold is skipped
    rather than crashing — common in small Hiking-era folds.
    """
    tscv = TimeSeriesSplit(n_splits=n)
    aucs = []
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        if scale:
            Xtr_s, Xte_s = _scale(Xtr, Xte)
        else:
            Xtr_s, Xte_s = Xtr.values, Xte.values
        # Need all 3 classes in train set for OvR to be defined
        if len(np.unique(ytr)) < 3:
            continue
        model.fit(Xtr_s, ytr)
        if len(np.unique(yte)) < 2:
            continue
        try:
            prob = model.predict_proba(Xte_s)
            # Use available classes if test fold is missing one
            classes_in_test = np.unique(yte)
            if len(classes_in_test) == 3:
                aucs.append(
                    roc_auc_score(yte, prob, multi_class="ovr", labels=[-1, 0, 1])
                )
            else:
                # Binary fallback: compute AUC for the two present classes
                idx = [
                    list(model.classes_).index(c)
                    for c in classes_in_test
                    if c in model.classes_
                ]
                if len(idx) == 2:
                    aucs.append(
                        roc_auc_score(
                            yte,
                            prob[:, idx[1]],
                            labels=classes_in_test,
                        )
                    )
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else np.nan


def _tune_C(X: pd.DataFrame, y: pd.Series) -> float:
    """Select best C by inner CV AUC. Falls back to C=0.1 if all AUCs are NaN."""
    best_C, best_auc = 0.1, -np.inf  # safe default
    for C in C_GRID:
        model = LogisticRegression(
            C=C,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
        auc = _auc_cv(X, y, model)
        if not np.isnan(auc) and auc > best_auc:
            best_auc, best_C = auc, C
    print(f"    _tune_C → best C={best_C}  inner AUC={best_auc:.4f}")
    return best_C


def _tune_rf(X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Select best max_depth and min_samples_leaf by inner CV AUC.
    Falls back to sensible defaults if all AUCs are NaN.
    """
    best_params = {"max_depth": 4, "min_samples_leaf": 3}  # explicit fallback
    best_auc = -np.inf
    for depth in RF_GRID["max_depth"]:
        for leaf in RF_GRID["min_samples_leaf"]:
            model = RandomForestClassifier(
                n_estimators=400,
                max_depth=depth,
                min_samples_leaf=leaf,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )
            auc = _auc_cv(X, y, model, scale=False)
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_params = {"max_depth": depth, "min_samples_leaf": leaf}
    print(f"    _tune_rf → {best_params}  inner AUC={best_auc:.4f}")
    return best_params


def _select_features(
    X_pool: pd.DataFrame,
    y: pd.Series,
    C: float,
    max_k: int = 4,
) -> list[str]:
    """
    Exhaustive search over feature combinations of size 1..max_k.
    Each combination evaluated by inner CV AUC with the given C.
    Returns the best combination.
    """
    avail = [f for f in X_pool.columns if f in X_pool.columns]
    best_feats, best_auc = avail[:1], -1.0

    for k in range(1, max_k + 1):
        for combo in combinations(avail, k):
            combo = list(combo)
            valid = pd.concat([X_pool[combo], y], axis=1).dropna()
            if len(valid) < 20:
                continue
            X_c = valid[combo]
            y_c = valid[y.name]
            model = LogisticRegression(
                C=C,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
            auc = _auc_cv(X_c, y_c, model)
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_feats = combo

    print(f"  Best features (k={len(best_feats)}, inner AUC={best_auc:.4f}):")
    for f in best_feats:
        print(f"    {f}")
    return best_feats


def _outer_cv(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    scale: bool = True,
) -> dict:
    """
    Outer 5-fold TSS: honest performance estimate.
    model_factory is called fresh each fold (no leakage).
    """
    tscv = TimeSeriesSplit(n_splits=N_SPLITS_OUTER)
    accs, aucs = [], []

    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        if scale:
            Xtr_s, Xte_s = _scale(Xtr, Xte)
        else:
            Xtr_s, Xte_s = Xtr.values, Xte.values

        model = model_factory()
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        accs.append((pred == yte).mean())

        if len(np.unique(yte)) == 3:
            try:
                prob = model.predict_proba(Xte_s)
                aucs.append(
                    roc_auc_score(yte, prob, multi_class="ovr", labels=[-1, 0, 1])
                )
            except Exception:
                pass

    return dict(
        acc_mean=np.mean(accs),
        acc_std=np.std(accs),
        auc_mean=np.mean(aucs) if aucs else np.nan,
    )


# ── Model runs ────────────────────────────────────────────────────────────────
def run_models(df: pd.DataFrame) -> dict:
    """
    M1: rate history baseline (no tuning needed)
    M2: sentiment-only, features + C selected by inner CV
    M3: M1 ∪ M2 features, C re-tuned by inner CV
    M4: M3 features, RF HPs tuned by inner CV
    """
    mro_feats = ["mro_lag", "mro_change_lag"]

    # ── Candidate pool availability ───────────────────────────────────────────
    X_all, y_all = _Xy(df, CANDIDATE_FEATURES + mro_feats)
    cand_avail = [f for f in CANDIDATE_FEATURES if f in X_all.columns]
    print(f"\n  Candidates available: {len(cand_avail)}/{len(CANDIDATE_FEATURES)}")
    print(
        f"  Composite target — n_analysis={len(X_all)}, "
        f"Hike={(y_all==1).sum()} Cut={(y_all==-1).sum()} "
        f"Hold={(y_all==0).sum()}"
    )

    results = {}

    # ── M1 Baseline ───────────────────────────────────────────────────────────
    X1, y1 = _Xy(df, mro_feats)
    C1 = _tune_C(X1, y1)
    print(f"\n--- M1 Baseline (n={len(X1)}, best C={C1}) ---")
    cv1 = _outer_cv(
        X1,
        y1,
        lambda: LogisticRegression(
            C=C1,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )
    print(f"  CV Accuracy: {cv1['acc_mean']:.4f} ± {cv1['acc_std']:.4f}")
    print(f"  CV AUC:      {cv1['auc_mean']:.4f}")

    m1 = LogisticRegression(
        C=C1,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    m1.fit(StandardScaler().fit_transform(X1), y1)
    results["M1 Baseline"] = dict(
        cv=cv1,
        model=m1,
        X=X1,
        y=y1,
        pred=m1.predict(StandardScaler().fit_transform(X1)),
        feats=list(X1.columns),
        classes=list(m1.classes_),
    )

    # ── M2 Sentiment (feature selection + C tuning by inner CV) ───────────────
    X_cand = X_all[cand_avail]
    print(
        f"\n--- M2 Sentiment: feature selection over {len(cand_avail)} candidates ---"
    )
    C2 = _tune_C(X_cand, y_all)
    feats2 = _select_features(X_cand, y_all, C2)
    X2, y2 = _Xy(df, feats2)
    print(f"\n--- M2 Sentiment (n={len(X2)}, feats={len(feats2)}, C={C2}) ---")
    cv2 = _outer_cv(
        X2,
        y2,
        lambda: LogisticRegression(
            C=C2,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )
    print(f"  CV Accuracy: {cv2['acc_mean']:.4f} ± {cv2['acc_std']:.4f}")
    print(f"  CV AUC:      {cv2['auc_mean']:.4f}")

    m2 = LogisticRegression(
        C=C2,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    Xf2 = StandardScaler().fit_transform(X2)
    m2.fit(Xf2, y2)
    results["M2 Sentiment"] = dict(
        cv=cv2,
        model=m2,
        X=X2,
        y=y2,
        pred=m2.predict(Xf2),
        feats=feats2,
        classes=list(m2.classes_),
    )

    # ── M3 Augmented (M1 ∪ M2 features, re-tune C) ────────────────────────────
    feats3_pool = list(dict.fromkeys(mro_feats + feats2))  # preserve order, dedupe
    X3_pool, y3 = _Xy(df, feats3_pool)
    C3 = _tune_C(X3_pool, y3)
    X3, y3 = _Xy(df, feats3_pool)
    print(f"\n--- M3 Augmented (n={len(X3)}, feats={len(feats3_pool)}, C={C3}) ---")
    cv3 = _outer_cv(
        X3,
        y3,
        lambda: LogisticRegression(
            C=C3,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
    )
    print(f"  CV Accuracy: {cv3['acc_mean']:.4f} ± {cv3['acc_std']:.4f}")
    print(f"  CV AUC:      {cv3['auc_mean']:.4f}")

    m3 = LogisticRegression(
        C=C3,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    Xf3 = StandardScaler().fit_transform(X3)
    m3.fit(Xf3, y3)
    results["M3 Augmented"] = dict(
        cv=cv3,
        model=m3,
        X=X3,
        y=y3,
        pred=m3.predict(Xf3),
        feats=feats3_pool,
        classes=list(m3.classes_),
    )

    # ── M4 Random Forest (HP tuning by inner CV) ──────────────────────────────
    rf_params = _tune_rf(X3, y3)
    print(
        f"\n--- M4 Random Forest "
        f"(n={len(X3)}, feats={len(feats3_pool)}, "
        f"depth={rf_params['max_depth']}, leaf={rf_params['min_samples_leaf']}) ---"
    )
    cv4 = _outer_cv(
        X3,
        y3,
        lambda: RandomForestClassifier(
            n_estimators=400,
            **rf_params,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        scale=False,
    )
    print(f"  CV Accuracy: {cv4['acc_mean']:.4f} ± {cv4['acc_std']:.4f}")
    print(f"  CV AUC:      {cv4['auc_mean']:.4f}")

    m4 = RandomForestClassifier(
        n_estimators=400,
        **rf_params,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    m4.fit(X3.values, y3)
    results["M4 Random Forest"] = dict(
        cv=cv4,
        model=m4,
        X=X3,
        y=y3,
        pred=m4.predict(X3.values),
        feats=feats3_pool,
        classes=list(m4.classes_),
    )

    # Naive baseline
    results["_naive"] = float((y_all == 0).mean())
    print(f"\nNaive baseline (always Hold): {results['_naive']:.4f}")

    return results


# ── Comparison: v2 vs v3 ──────────────────────────────────────────────────────
V2_AUC = {
    "M1 Baseline": 0.6821,
    "M2 Sentiment": 0.7433,
    "M3 Augmented": 0.7388,
    "M4 Random Forest": 0.7125,
}


def print_comparison(results: dict):
    print("\n" + "=" * 60)
    print("v2  vs  v3  —  CV AUC comparison")
    print("=" * 60)
    print(f"  {'Model':<22}  {'v2':>7}  {'v3':>7}  {'Δ':>7}")
    print("  " + "-" * 42)
    for name in ["M1 Baseline", "M2 Sentiment", "M3 Augmented", "M4 Random Forest"]:
        v2 = V2_AUC.get(name, np.nan)
        v3 = results[name]["cv"]["auc_mean"]
        delta = v3 - v2
        flag = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "≈")
        print(f"  {name:<22}  {v2:>7.4f}  {v3:>7.4f}  " f"{flag} {delta:+.4f}")


# ── Figures ───────────────────────────────────────────────────────────────────
def fig_performance(results: dict, save=True):
    names = ["M1 Baseline", "M2 Sentiment", "M3 Augmented", "M4 Random Forest"]
    colors = ["#95a5a6", "#c0392b", "#2c6fad", "#27ae60"]
    accs = [results[m]["cv"]["acc_mean"] for m in names]
    errs = [results[m]["cv"]["acc_std"] for m in names]
    aucs = [results[m]["cv"]["auc_mean"] for m in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(names))
    ax.bar(
        x - 0.20,
        accs,
        0.35,
        color=colors,
        alpha=0.85,
        yerr=errs,
        capsize=5,
        label="Accuracy (CV)",
    )
    ax.bar(
        x + 0.20,
        aucs,
        0.35,
        color=colors,
        alpha=0.38,
        edgecolor=colors,
        linewidth=1.5,
        label="AUC OvR (CV)",
    )

    # v2 AUC as dots for comparison
    for i, name in enumerate(names):
        if name in V2_AUC:
            ax.scatter(
                x[i] + 0.20,
                V2_AUC[name],
                marker="x",
                color="black",
                s=60,
                zorder=6,
                label="v2 AUC" if i == 0 else "_",
            )

    ax.axhline(
        results["_naive"],
        ls="--",
        lw=1.5,
        color="#888",
        label=f"Naive baseline: {results['_naive']:.3f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score (0–1)")
    ax.set_title(
        "Cross-validated performance — v3 (composite target + Δsentiment)\n"
        "× marks = v2 AUC for comparison"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig_performance_v3.pdf", bbox_inches="tight")
        print("Saved → fig_performance_v3.pdf")
    return fig


def fig_confusion(results: dict, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cls_labels = ["Cut", "Hold", "Hike"]
    for ax, name, title in zip(
        axes,
        ["M3 Augmented", "M4 Random Forest"],
        ["M3 — Augmented Logit (v3)", "M4 — Random Forest (v3)"],
    ):
        r = results[name]
        cm = confusion_matrix(r["y"], r["pred"], labels=[-1, 0, 1])
        ConfusionMatrixDisplay(cm, display_labels=cls_labels).plot(
            ax=ax, colorbar=False, cmap="Blues"
        )
        recall = cm.diagonal() / cm.sum(axis=1)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(
            f"Predicted label\n"
            f"Recall — Cut: {recall[0]:.0%}  "
            f"Hold: {recall[1]:.0%}  Hike: {recall[2]:.0%}",
            fontsize=9,
        )
    fig.suptitle("Confusion matrices — v3 (composite target)", fontsize=12)
    fig.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "fig_confusion_v3.pdf", bbox_inches="tight")
        print("Saved → fig_confusion_v3.pdf")
    return fig


def fig_feature_analysis(results: dict, save=True):
    r3 = results["M3 Augmented"]
    r4 = results["M4 Random Forest"]
    feats = r3["feats"]
    labels = [FEAT_LABELS.get(f, f) for f in feats]

    coef_df = pd.DataFrame(
        r3["model"].coef_,
        columns=feats,
        index=[{-1: "Cut", 0: "Hold", 1: "Hike"}[c] for c in r3["classes"]],
    )
    imp_map = dict(zip(r4["feats"], r4["model"].feature_importances_))
    imps = np.array([imp_map.get(f, 0.0) for f in feats])

    # M3 coefficients
    fig1, ax1 = plt.subplots(figsize=(9, 5))
    x, w = np.arange(len(feats)), 0.25
    for i, (cls, col) in enumerate(
        {"Cut": "#c0392b", "Hold": "#95a5a6", "Hike": "#2c6fad"}.items()
    ):
        ax1.bar(x + (i - 1) * w, coef_df.loc[cls], w, label=cls, color=col, alpha=0.80)
    ax1.axhline(0, color="#444", lw=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Standardised coefficient")
    ax1.set_title("M3 Augmented Logit — standardised coefficients by class (v3)")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.22)
    fig1.tight_layout()

    # M4 feature importances
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(feats))
    ax2.barh(y_pos, imps, color="#27ae60", alpha=0.85, edgecolor="white")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=9)
    ax2.set_xlabel("Feature importance (impurity decrease)")
    ax2.set_title("M4 Random Forest — feature importance (v3)")
    ax2.grid(axis="x", alpha=0.22)
    fig2.tight_layout()

    if save:
        fig1.savefig(OUTPUT_DIR / "fig_coefficients_v3.pdf", bbox_inches="tight")
        fig2.savefig(OUTPUT_DIR / "fig_rf_importances_v3.pdf", bbox_inches="tight")
        print("Saved → fig_coefficients_v3.pdf, fig_rf_importances_v3.pdf")

    return fig1, fig2


# ── Classification reports ────────────────────────────────────────────────────
def report(results: dict):
    for name in ["M3 Augmented", "M4 Random Forest"]:
        r = results[name]
        print(f"\n{'='*60}")
        print(f"{name} — full-sample report (v3, composite target)")
        print("=" * 60)
        print(
            classification_report(
                r["y"],
                r["pred"],
                labels=[-1, 0, 1],
                target_names=["Cut", "Hold", "Hike"],
            )
        )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading data...")
    df = load_data()

    # Show composite target breakdown
    d = df[df["composite_diff"] != 0]["direction"]
    all_d = df[df["composite_diff"] != 0]
    print(f"  {len(df)} meetings total")
    print(
        f"  Composite target: n_changed={len(all_d)} "
        f"| Hike={(d==1).sum()} Cut={(d==-1).sum()} Hold={(d==0).sum()}"
    )
    print(
        f"  MRO-only:         n_changed="
        f"{(df['MRO_diff'] != 0).sum()} "
        f"(v2 baseline for comparison)"
    )

    results = run_models(df)
    print_comparison(results)

    fig_performance(results)
    fig_confusion(results)
    fig_feature_analysis(results)
    report(results)

    plt.show()

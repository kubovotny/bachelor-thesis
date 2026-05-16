"""
Section 3.9 – Predicting the Direction of ECB Rate Changes

Logistic regression (multinomial) predicting MRO decision direction:
  +1 = Hike  (ΔMRO > 0)
  -1 = Cut   (ΔMRO < 0)
   0 = Hold  (ΔMRO = 0)

Three models:
  M1 (Baseline) : features = [MRO_lag]
  M2 (Sentiment): features = [Δsentiment features, sentiment_std]
  M3 (Augmented): features = [MRO_lag + sentiment features]

Evaluation: TimeSeriesSplit accuracy + confusion matrix + AUC (OvR)

Supervisor note (29.4): "difference in sentiment is more important than
level" → use Δsentiment as primary features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, classification_report,
                              roc_auc_score, ConfusionMatrixDisplay)

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/model"
OUTPUT_DIR.mkdir(exist_ok=True)

WORD_LIMIT = 1   # single sentences — best per Section 2.X analysis
N_SPLITS   = 5

# Features: top 4 predictors from Section 3.6 lag analysis
# Deliberately parsimonious (n_changed ≈ 56 → avoid overfitting)
# Note: lag choice informed by correlation analysis (acknowledged in text)
SENT_FEATS = [
    "finbert_mean_lag2",    # peak lag for full sample (r=0.50***)
    "finbert_mean_lag5",    # second peak (r=0.56***)
    "finbert_IS_mean_lag2", # IS-specific signal (r=0.84*** Pre-ZLB)
    "finbert_IS_std",       # volatility — communication consistency
]


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    configs = [
        dict(IS_QA_division=False, qa_options="both_together", with_label=False),
        dict(IS_QA_division=True,  qa_options="just_answers",  with_label=False),
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

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Target: direction of MRO change (use MRO.diff() per notebook finding)
    df["MRO_diff"] = df["MRO"].diff()
    df["direction"] = np.sign(df["MRO_diff"]).fillna(0).astype(int)

    # Lag features for all sentiment columns
    sent_cols = [c for c in df.columns
                 if any(m in c for m in ["finbert", "roberta"])
                 and c.endswith(("_mean", "_std", "_max", "_min"))]
    for col in sent_cols:
        for lag in [1, 2, 3, 5, 7, 8, 11, 13]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df["mro_lag"]        = df["MRO"].shift(1)
    df["mro_change_lag"] = df["MRO_diff"].shift(1)

    return df


def build_Xy(df: pd.DataFrame, feature_set: list) -> tuple:
    cols = feature_set + ["direction"]
    clean = df[cols].dropna()
    X = clean[feature_set]
    y = clean["direction"]
    return X, y, clean.index


def scale(X_train, X_test):
    sc = StandardScaler()
    return sc.fit_transform(X_train), sc.transform(X_test)


# ── Cross-validated accuracy ──────────────────────────────────────────────────
def cv_accuracy(X: pd.DataFrame, y: pd.Series,
                model, n_splits: int = N_SPLITS) -> dict:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs, aucs = [], []

    for train_idx, test_idx in tscv.split(X):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        X_tr_sc, X_te_sc = scale(X_tr, X_te)

        model.fit(X_tr_sc, y_tr)
        pred = model.predict(X_te_sc)
        accs.append((pred == y_te).mean())

        # AUC — only if all 3 classes present in test
        if len(np.unique(y_te)) == 3:
            prob = model.predict_proba(X_te_sc)
            try:
                auc = roc_auc_score(y_te, prob, multi_class="ovr",
                                    labels=[-1, 0, 1])
                aucs.append(auc)
            except Exception:
                pass

    return {
        "acc_mean": np.mean(accs),
        "acc_std":  np.std(accs),
        "auc_mean": np.mean(aucs) if aucs else np.nan,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def run_models(df: pd.DataFrame) -> dict:
    # Use lag features directly (already computed in load_data)
    sent_features = [f for f in SENT_FEATS if f in df.columns]
    print(f"  Available sentiment features: {len(sent_features)}/{len(SENT_FEATS)}")
    print(f"  Features: {sent_features}")

    feature_sets = {
        "M1 Baseline":   ["mro_lag", "mro_change_lag"],
        "M2 Sentiment":  sent_features,
        "M3 Augmented":  ["mro_lag", "mro_change_lag"] + sent_features,
    }

    clf = LogisticRegression(
        class_weight="balanced", solver="lbfgs",
        max_iter=2000, C=1.0, random_state=42,
    )

    results = {}
    print(f"\nClass distribution: {dict(df['direction'].value_counts().sort_index())}")

    for name, feats in feature_sets.items():
        feats_avail = [f for f in feats if f in df.columns]
        X, y, idx = build_Xy(df, feats_avail)
        print(f"\n--- {name} (n={len(X)}, features={len(feats_avail)}) ---")

        cv = cv_accuracy(X, y, clf)
        print(f"  CV Accuracy: {cv['acc_mean']:.4f} ± {cv['acc_std']:.4f}")
        print(f"  CV AUC (OvR): {cv['auc_mean']:.4f}" if not np.isnan(cv['auc_mean']) else "  AUC: N/A")

        # Full-sample fit for confusion matrix + coefficients
        sc = StandardScaler()
        X_sc = sc.fit_transform(X)
        clf.fit(X_sc, y)
        pred = clf.predict(X_sc)

        results[name] = {
            "cv": cv, "model": clf, "X": X, "y": y,
            "pred": pred, "features": feats_avail,
            "classes": clf.classes_,
        }

    # Baseline: always predict "Hold"
    _, y_all, _ = build_Xy(df, ["mro_lag"])
    hold_acc = (y_all == 0).mean()
    print(f"\nNaive baseline (always Hold): {hold_acc:.4f}")

    results["_naive_acc"] = hold_acc
    return results


# ── Figure ────────────────────────────────────────────────────────────────────
def plot_fig_3_9(results: dict, save: bool = True):
    """
    Upravená verzia: Rozdelí vizualizáciu na 3 samostatné grafy
    a detailný report vypíše priamo do terminálu.
    """
    model_names = ["M1 Baseline", "M2 Sentiment", "M3 Augmented"]
    colors      = ["#95a5a6", "#c0392b", "#2c6fad"]

    # ── GRAF 1: Úspešnosť modelov (Accuracy & AUC) ──────────────────────────
    # Toto nám hovorí, ako dobre náš "robot" predpovedá realitu.
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    x = np.arange(len(model_names))
    accs = [results[m]["cv"]["acc_mean"] for m in model_names]
    errs = [results[m]["cv"]["acc_std"]  for m in model_names]
    aucs = [results[m]["cv"]["auc_mean"] for m in model_names]

    ax1.bar(x - 0.2, accs, 0.35, color=colors, alpha=0.85, yerr=errs, capsize=5, label="Presnosť (Accuracy)")
    ax1.bar(x + 0.2, aucs, 0.35, color=colors, alpha=0.45, edgecolor=colors, linewidth=1.5, label="Spoľahlivosť (AUC)")

    naive = results["_naive_acc"]
    ax1.axhline(naive, color="#888", linewidth=1.5, linestyle="--", label=f"Naivný odhad (vždy Hold): {naive:.3f}")
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.set_ylabel("Skóre (0 až 1)")
    ax1.set_title("1. Porovnanie úspešnosti modelov")
    ax1.legend(loc="lower right")
    ax1.grid(axis="y", alpha=0.25)

    if save:
        fig1.savefig(OUTPUT_DIR / "fig_performance.pdf", bbox_inches="tight")

    # ── GRAF 2: Matica zámen (Kde si model mýli pojmy) ───────────────────────
    # Ukazuje nám "mapu chýb" – napríklad či model čakal zvýšenie, keď prišlo zníženie.
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    r = results["M3 Augmented"]
    cm = confusion_matrix(r["y"], r["pred"], labels=[-1, 0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Zníženie (Cut)", "Ponechanie (Hold)", "Zvýšenie (Hike)"])
    disp.plot(ax=ax2, colorbar=False, cmap="Blues")
    ax2.set_title("2. Matica chýb modelu M3 (Confusion Matrix)")

    if save:
        fig2.savefig(OUTPUT_DIR / "fig_confusion.pdf", bbox_inches="tight")

    # ── GRAF 3: Dôležitosť indícií (Koeficienty) ────────────────────────────
    # Ktoré slová alebo údaje mali na model najväčší vplyv?
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    m3 = results["M3 Augmented"]["model"]
    feats = results["M3 Augmented"]["features"]
    classes = results["M3 Augmented"]["classes"]
    coef_df = pd.DataFrame(m3.coef_, columns=feats, index=[f"Trieda {c}" for c in classes]).T

    x_pos = np.arange(len(feats))
    width = 0.25
    cls_colors = ["#c0392b", "#95a5a6", "#2c6fad"]
    for i, (cls, col) in enumerate(zip(coef_df.columns, cls_colors)):
        offset = (i - 1) * width
        ax3.bar(x_pos + offset, coef_df[cls], width, label=cls, color=col, alpha=0.75)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f[:20] for f in feats], rotation=35, ha="right")
    ax3.axhline(0, color="#555", linewidth=0.8)
    ax3.set_ylabel("Sila vplyvu (Koeficient)")
    ax3.set_title("3. Čo najviac ovplyvnilo rozhodovanie modelu")
    ax3.legend()
    ax3.grid(axis="y", alpha=0.22)

    if save:
        fig3.savefig(OUTPUT_DIR / "fig_coefficients.pdf", bbox_inches="tight")

    # ── VÝSTUP DO KONZOLY: Klasifikačný report ──────────────────────────────
    # Toto je "vysvedčenie" nášho najlepšieho modelu.
    r3 = results["M3 Augmented"]
    rep = classification_report(r3["y"], r3["pred"],
                                 labels=[-1, 0, 1],
                                 target_names=["Zníženie (Cut)", "Ponechanie (Hold)", "Zvýšenie (Hike)"])
    
    print("\n" + "="*60)
    print("DETAILNÝ REPORT MODELU M3 (Augmented)")
    print("="*60)
    print(rep)
    print("="*60 + "\n")

    return [fig1, fig2, fig3]


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} meetings | Hike: {(df['direction']==1).sum()} | "
          f"Cut: {(df['direction']==-1).sum()} | "
          f"Hold: {(df['direction']==0).sum()}")

    results = run_models(df)
    plot_fig_3_9(results)
    plt.show()
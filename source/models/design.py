import re


def make_pretty(feature: str) -> str:
    """
    Convert internal feature names to readable axis labels.

    Examples
    --------
    finbert_mean_lag4        →  FB μ [t−4]
    finbert_IS_max_lag2      →  FB IS max [t−2]
    finbert_IS_MP_mean_lag2  →  FB IS·MP μ [t−2]
    finbert_IS_std           →  FB IS σ
    roberta_QA_EP_mean_lag7  →  RB QA·EP μ [t−7]
    mro_lag                  →  MRO [t−1]
    mro_change_lag           →  ΔMRO [t−1]
    d_finbert_mean_lag4      →  ΔFB μ [t−4]
    """
    # ── rate history features ─────────────────────────────────────────────
    if feature == "mro_lag":
        return "MRO [t−1]"
    if feature == "mro_change_lag":
        return "ΔMRO [t−1]"

    # ── delta prefix ──────────────────────────────────────────────────────
    delta = ""
    if feature.startswith("d_"):
        delta = "Δ"
        feature = feature[2:]

    # ── lag suffix ────────────────────────────────────────────────────────
    lag_str = ""
    lag_match = re.search(r"_lag(\d+)$", feature)
    if lag_match:
        lag_str = f" [t−{lag_match.group(1)}]"
        feature = feature[: lag_match.start()]

    # ── model prefix ──────────────────────────────────────────────────────
    if feature.startswith("finbert_"):
        model = "FB"
        feature = feature[len("finbert_") :]
    elif feature.startswith("roberta_"):
        model = "RB"
        feature = feature[len("roberta_") :]
    else:
        model = feature
        feature = ""

    # ── section token (IS / QA) ───────────────────────────────────────────
    section = ""
    for tok in ("IS", "QA"):
        if feature.startswith(tok + "_"):
            section = tok
            feature = feature[len(tok) + 1 :]
            break

    # ── topic token (MP / EP / FS / OI) ──────────────────────────────────
    topic = ""
    for tok in ("MP", "EP", "FS", "OI"):
        if feature.startswith(tok + "_"):
            topic = tok
            feature = feature[len(tok) + 1 :]
            break

    # ── statistic token ───────────────────────────────────────────────────
    STAT_MAP = {
        "mean": "μ",
        "std": "σ",
        "max": "max",
        "min": "min",
    }
    stat = STAT_MAP.get(feature, feature)

    # ── assemble ──────────────────────────────────────────────────────────
    section_topic = ""
    if section and topic:
        section_topic = f" {section}·{topic}"
    elif section:
        section_topic = f" {section}"
    elif topic:
        section_topic = f" {topic}"

    return f"{delta}{model}{section_topic} {stat}{lag_str}".strip()


if __name__ == "__main__":
    tests = [
        "finbert_mean_lag",
        "finbert_mean_lag2",
        "finbert_IS_max_lag2",
        "finbert_IS_std",
        "finbert_IS_MP_mean_lag2",
        "roberta_IS_mean_lag9",
        "roberta_QA_EP_mean_lag7",
        "d_finbert_mean_lag4",
        "mro_lag",
        "mro_change_lag",
    ]

    for t in tests:
        print(f"{t:35s}  →  {make_pretty(t)}")

# local_projections.py
import pandas as pd, numpy as np, statsmodels.api as sm
from ..data.model_data import return_data
import matplotlib.pyplot as plt

df = return_data(market_data="all", word_limit=200, qa_options="just_answers",
                 with_label=False, IS_QA_division=True).sort_values("date").reset_index(drop=True)

shock = "roberta_QA_mean"          # your headline sentiment surprise
target = "OIS_1Y"                  # 1-year OIS rate
horizons = range(0, 11)            # h = 0 .. 10 conferences ahead


# Build lead targets
for h in horizons:
    df[f"y_h{h}"] = 100 * (df[target].shift(-h) - df[target])   # cumulative change

# Controls: lagged target + lagged shock (Plagborg-Møller & Wolf 2021 recommend 2 lags)
controls = []
for lag in [1, 2]:
    df[f"{target}_l{lag}"] = df[target].shift(lag)
    df[f"{shock}_l{lag}"]  = df[shock].shift(lag)
    controls += [f"{target}_l{lag}", f"{shock}_l{lag}"]

coefs, lo, hi = [], [], []
for h in horizons:
    sub = df.dropna(subset=[f"y_h{h}", shock] + controls)
    sub[shock] = (sub[shock] - sub[shock].mean()) / sub[shock].std()
    X = sm.add_constant(sub[[shock] + controls])
    res = sm.OLS(sub[f"y_h{h}"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    b = res.params[shock]; se = res.bse[shock]
    coefs.append(b); lo.append(b - 1.96*se); hi.append(b + 1.96*se)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(list(horizons), coefs, color="black")
ax.fill_between(list(horizons), lo, hi, alpha=0.25)
ax.axhline(0, lw=0.8, color="grey")
ax.set_xlabel("Horizon (press conferences ahead)")
ax.set_ylabel(f"Cumulative response of {target} (pp)")
ax.set_title(f"Local-projection IRF: shock = {shock}")
fig.tight_layout()
plt.show()
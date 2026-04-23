from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sentiment import return_sentiment_data
from market import return_market_data
import pandas as pd

data = pd.merge(
    return_market_data("shocks_ecb_mpd_me_d.csv"), return_sentiment_data("roberta", with_label=True), on="date"
)
data

treshold = data["pc1"].quantile(0.7)
data["to_invest"] = data["pc1"] > treshold
print(data.columns)
X_COLUMNS = ['mean_IS_EP',
       'mean_IS_FS', 'mean_IS_MP', 'mean_IS_OI', 'mean_QA_EP', 'mean_QA_FS',
       'mean_QA_MP', 'mean_QA_OI', 'max_IS_EP', 'max_IS_FS', 'max_IS_MP',
       'max_IS_OI', 'max_QA_EP', 'max_QA_FS', 'max_QA_MP', 'max_QA_OI',
       'min_IS_EP', 'min_IS_FS', 'min_IS_MP', 'min_IS_OI', 'min_QA_EP',
       'min_QA_FS', 'min_QA_MP', 'min_QA_OI', 'std_IS_EP', 'std_IS_FS',
       'std_IS_MP', 'std_IS_OI', 'std_QA_EP', 'std_QA_FS', 'std_QA_MP',
       'std_QA_OI']
# X_COLUMNS = [f"{a}_{b}" for a in ["mean", "std", "max", "min"] for b in ["IS", "QA"]]

# data = data.query("date < '2010-01-01'")


# 1. Rozdelenie dát (stále zachovávame časovú os!)
X_train, X_test, y_train, y_test = train_test_split(
    data[X_COLUMNS], data["to_invest"], test_size=0.2, shuffle=False
)
print(data.columns)
# 3. Škálovanie
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- MODEL 1: Logistická Regresia ---
log_model = LogisticRegression(class_weight='balanced', random_state=42)
log_model.fit(X_train_scaled, y_train)
auc_log = roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:, 1])

# --- MODEL 2: Random Forest ---
rf_model = RandomForestClassifier(n_estimators=200, max_depth=4, class_weight='balanced_subsample', random_state=42)
rf_model.fit(X_train, y_train) # RF nepotrebuje nutne škálovanie
auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

print(f"--- VÝSLEDKY S REDUKOVANÝMI PREMENNÝMI (8 stĺpcov) ---")
print(f"Logistická Regresia AUC: {auc_log:.3f}")
print(f"Random Forest AUC: {auc_rf:.3f}")

# Pozrime sa, čo je pre model teraz najdôležitejšie
importances = pd.DataFrame({
    "Premenna": X_COLUMNS, 
    "Dolezitost": rf_model.feature_importances_
}).sort_values(by="Dolezitost", ascending=False)

print("\n--- TOP PREMENNÉ (Random Forest) ---")
print(importances[:10])
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from ..data.model_data import return_data
import pandas as pd

# FIXME calling random forest
# TODO tune model

data = return_data("roberta", with_label=False)
treshold = data["pc1"].quantile(0.7)
data["to_invest"] = data["pc1"] > treshold
X_COLUMNS = ["mean_IS_EP",
       "mean_IS_FS", "mean_IS_MP", "mean_IS_OI", "mean_QA_EP", "mean_QA_FS",
       "mean_QA_MP", "mean_QA_OI", "max_IS_EP", "max_IS_FS", "max_IS_MP",
       "max_IS_OI", "max_QA_EP", "max_QA_FS", "max_QA_MP", "max_QA_OI",
       "min_IS_EP", "min_IS_FS", "min_IS_MP", "min_IS_OI", "min_QA_EP",
       "min_QA_FS", "min_QA_MP", "min_QA_OI", "std_IS_EP", "std_IS_FS",
       "std_IS_MP", "std_IS_OI", "std_QA_EP", "std_QA_FS", "std_QA_MP",
       "std_QA_OI"]
X_COLUMNS = [f"{a}_{b}" for a in ["mean", "std", "max", "min"] for b in ["IS", "QA"]]
print(X_COLUMNS)
# data = data.query("date < "2010-01-01"")


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
log_model = LogisticRegression(class_weight="balanced", random_state=42)
log_model.fit(X_train_scaled, y_train)
auc_log = roc_auc_score(y_test, log_model.predict_proba(X_test_scaled)[:, 1])

# --- MODEL 2: Random Forest ---
rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=4, class_weight="balanced_subsample", random_state=42
)
rf_model.fit(X_train, y_train)  # RF nepotrebuje nutne škálovanie
y_pred = rf_model.predict(X_test)
auc_rf = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

print(f"--- VÝSLEDKY S REDUKOVANÝMI PREMENNÝMI (8 stĺpcov) ---")
print(f"Logistická Regresia AUC: {auc_log:.3f}")
print(f"Random Forest AUC: {auc_rf:.3f}")

# Pozrime sa, čo je pre model teraz najdôležitejšie
importances = pd.DataFrame(
    {"Premenna": X_COLUMNS, "Dolezitost": rf_model.feature_importances_}
).sort_values(by="Dolezitost", ascending=False)

print("\n--- TOP PREMENNÉ (Random Forest) ---")
print(importances[:10])

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

# y_test sú tvoje reálne dáta, y_pred sú predikcie (0 alebo 1)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# 2. Získanie tvrdých predikcií (Pre Precision, Recall a Confusion Matrix)
y_pred = rf_model.predict(X_test)

# 3. Vypísanie metrík (Zadanie pre Vinářa)
from sklearn.metrics import classification_report, confusion_matrix

print("--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred))

print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred))

precision, recall, _ = precision_recall_curve(y_test, y_prob_rf)
# print(precision, recall)

# Zoberieme mäkké pravdepodobnosti
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Vytvoríme vlastné y_pred s prísnejším/voľnejším prahom (napr. 0.35)
# Ak je pravdepodobnosť >= 0.35, vráti True (1), inak False (0)
vlastny_prah = 0.4
print(f"s prahom {vlastny_prah}:")
y_pred_custom = (y_prob_rf >= vlastny_prah).astype(int)

# Teraz pozeráš metriky pre TVOJ vlastný prah
print(classification_report(y_test, y_pred_custom))
print("--- CONFUSION MATRIX ---")
print(confusion_matrix(y_test, y_pred_custom))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Vypočítanie matice z tvojich existujúcich premenných
cm = confusion_matrix(y_test, y_pred_custom)

plt.figure(figsize=(8, 6))

# Vykreslenie teplotnej mapy (heatmap)
# cmap='Blues' urobí pekný modrý prechod, 'd' znamená celé čísla
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Bežný deň (0)', 'Trhový šok (1)'],
            yticklabels=['Bežný deň (0)', 'Trhový šok (1)'],
            annot_kws={"size": 18, "weight": "bold"}) # Veľké a tučné čísla

plt.title('Matica zámen (Confusion Matrix) - Vlastný prah', fontsize=16, pad=15)
plt.xlabel('Predikovaná trieda (Čo povedal model)', fontsize=14, labelpad=10)
plt.ylabel('Skutočná trieda (Čo sa reálne stalo)', fontsize=14, labelpad=10)

plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png", dpi=300)
plt.show()


# y_test = tvoje skutočné hodnoty (0 alebo 1)
# y_prob = tvoje predikované pravdepodobnosti z modelu (napr. RoBERTa)

# Vypočítame Precision a Recall pre všetky možné prahy (0.0 až 1.0)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_rf)

# Vypočítame F1-score (harmonický priemer Precision a Recall)
# Pridáme malú hodnotu (epsilon), aby sme predišli deleniu nulou
f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-10)

# Vyberieme si nejaký náš biznisový prah na ukážku (napr. 0.35)
CUSTOM_THRESHOLD = 0.4

# --- VYKRESLENIE ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# 1. PODGRAF: Distribúcia pravdepodobností (KDE Plot)
# Ukazuje, ako model rozdelil bežné dni (0) a dni so šokom (1)
sns.kdeplot(y_prob_rf[y_test == 0], fill=True, color="blue", label="Bežný deň (y=0)", ax=ax1, alpha=0.3)
sns.kdeplot(y_prob_rf[y_test == 1], fill=True, color="red", label="Trhový šok (y=1)", ax=ax1, alpha=0.3)

ax1.axvline(CUSTOM_THRESHOLD, color='black', linestyle='--', linewidth=2, label=f'Náš Prah ({CUSTOM_THRESHOLD})')
ax1.set_title("Rozdelenie pravdepodobností modelu", fontsize=14)
ax1.set_xlabel("Predikovaná pravdepodobnosť (0 až 1)")
ax1.set_ylabel("Hustota")
ax1.legend()

# 2. PODGRAF: Trade-off krivka (Ako prah mení metriky)
ax2.plot(thresholds, precisions[:-1], "b--", label="Precision (Presnosť pozitívnych)", linewidth=2)
ax2.plot(thresholds, recalls[:-1], "g-", label="Recall (Záchyt pozitívnych)", linewidth=2)
ax2.plot(thresholds, f1_scores, "r-", label="F1 Score", linewidth=2.5)

# Zvýraznenie nášho vybraného prahu na krivkách
ax2.axvline(CUSTOM_THRESHOLD, color='black', linestyle='--', linewidth=2)
ax2.set_title("Vplyv prahu na výkonnosť modelu (Threshold Tuning)", fontsize=14)
ax2.set_xlabel("Prah pravdepodobnosti (Threshold)")
ax2.set_ylabel("Skóre metrík (0 až 1)")
ax2.legend(loc="lower center")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("threshold_analysis.png", dpi=300)
plt.show()


# Predpokladám, že tvoj natrénovaný model sa volá rf_roberta 
# a trénovacie dáta boli v DataFrame X_train_R (s názvami stĺpcov)

# 1. Extrakcia dôležitosti z Random Forestu
importances = rf_model.feature_importances_
feature_names = X_train.columns

# 2. Vytvorenie DataFrame pre ľahšie zoradenie
fi_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Zoradenie od najdôležitejšej po najmenej dôležitú
fi_df = fi_df.sort_values(by='Importance', ascending=False)

# 3. Vykreslenie
plt.figure(figsize=(12, 7))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')

# Estetika
plt.title("Dôležitosť premenných: Čo najviac predikuje trhový šok? (RoBERTa)", fontsize=15)
plt.xlabel("Miera dôležitosti (Gini Importance)", fontsize=12)
plt.ylabel("Premenné z textu a trhu", fontsize=12)
plt.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=300)
plt.show()
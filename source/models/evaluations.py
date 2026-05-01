from sklearn.metrics import classification_report, confusion_matrix, average_precision_score

def evaluate_model(model_name, y_true, y_pred, y_prob):
    print(f"\n{'='*40}")
    print(f"VÝSLEDKY PRE MODEL: {model_name.upper()}")
    print(f"{'='*40}")
    
    # 1. PR-AUC (Oveľa lepšie ako ROC-AUC pre finančné šoky)
    pr_auc = average_precision_score(y_true, y_prob)
    print(f"PR-AUC (Average Precision): {pr_auc:.3f}\n")
    
    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"True Negatives: {cm[0][0]} | False Positives: {cm[0][1]} (Falošný poplach)")
    print(f"False Negatives: {cm[1][0]} (Zmeškaný šok) | True Positives: {cm[1][1]} (Zachytený šok)\n")
    
    # 3. Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

# Vyhodnotenie
# evaluate_model("RoBERTa", y_test, y_pred_R, y_prob_R)
# evaluate_model("FinBERT", y_test, y_pred_F, y_prob_F)
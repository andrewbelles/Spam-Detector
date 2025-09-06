#!/bin/python3 
# 
# inference.py  Andrew Belles  Sept 6th, 2025 
# 
# Performs inference on encoded test data using best svm model on file 
# 

import joblib, numpy as np
from text_encoding import load_samples
from sklearn.metrics import (
        classification_report, confusion_matrix,
        average_precision_score, roc_auc_score
)

def main():

    X, y = load_samples("test_samples.npz")
    model = joblib.load("svm_best.joblib")

    probs = model.predict_proba(X)[:, 1]

    pr_auc = average_precision_score(y, probs)
    roc_auc = roc_auc_score(y, probs)
    preds = (probs >= 0.5).astype(int)
    q = np.quantile(probs, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])

    print(f"Score quantiles: {np.round(q, 6)}")
    print(f"Predicted positive rate: {preds.mean():.3f}")
    print(f"PR-AUC: {pr_auc:.5f}")
    print(f"ROC-AUC: {roc_auc:0.5f}")
    print("\nClassification report (threshold=0.5):\n",
          classification_report(y, preds, target_names=["non-spam","spam"], digits=3))
    print("Confusion matrix:\n", confusion_matrix(y, preds))

    
if __name__ == "__main__":
    main()

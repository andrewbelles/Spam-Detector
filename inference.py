#!/bin/python 
# 
# inference.py  Andrew Belles  Sept 6th, 2025 
# 
# Performs inference on encoded test data using best svm model on file 
# 

import joblib, sys, csv
from text_encoding import load_samples
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt 

label_names = ["non-spam", "spam"] 

def main():
    ''' 
    Performs inference on test_samples and provides outputs to be shown to user. 
    '''

    # Load Data and Model 
    X, y = load_samples("test_samples.npz")
    model = joblib.load("svm_best.joblib")

    # Extract predictions 
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Print results to stdout 
    report = classification_report(y, preds, target_names=label_names, 
                                   digits=3, output_dict=True)

    writer = csv.writer(sys.stdout, delimiter="\t", lineterminator="\n")
    writer.writerow(["class", "precision", "recall", "f1", "support"])
    for name in label_names:
        r = report[name]
        writer.writerow([name, f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                         f"{r['f1-score']:.3f}", int(r['support'])])

# accuracy row
    total_support = sum(int(report[n]["support"]) for n in label_names)
    writer.writerow(["accuracy", "", "", f"{report['accuracy']:.3f}", total_support])

# macro/weighted
    for name in ["macro avg", "weighted avg"]:
        r = report[name]
        writer.writerow([name, f"{r['precision']:.3f}", f"{r['recall']:.3f}",
                         f"{r['f1-score']:.3f}", int(r['support'])])

    cm = confusion_matrix(y, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
    disp.plot(values_format="d", cmap="Blues")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()
    
if __name__ == "__main__":
    main()

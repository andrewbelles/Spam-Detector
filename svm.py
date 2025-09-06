#!/bin/python
# 
# svm.py  Andrew Belles  Sept 6th, 2025 
# Fits an SVM to training_samples produced by text_encoding file 
# 

import joblib, numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV 
from scipy.stats import loguniform 
from sklearn.calibration import CalibratedClassifierCV
from text_encoding import load_samples

def main():
    # Load training_samples 
    X, y = load_samples("training_samples.npz")

    svc = LinearSVC(class_weight="balanced", dual=False, loss="squared_hinge", penalty="l2")
    param_grid = {
        "C": loguniform(1e-3, 1e+1),
        "tol": [1e-4, 1e-3],
    }

    cross = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    search = RandomizedSearchCV(svc, param_distributions=param_grid, n_iter=30, scoring="average_precision", 
                                cv=cross, n_jobs=-1, random_state=0, verbose=51, refit=True)

    search.fit(X, y)
    best = search.best_estimator_

    calibrated = CalibratedClassifierCV(best, method="sigmoid", cv=5)
    calibrated.fit(X, y)

    joblib.dump(calibrated, "svm_best.joblib")


if __name__ == "__main__":
    main()

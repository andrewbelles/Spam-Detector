#!/bin/python
# 
# svm.py  Andrew Belles  Sept 6th, 2025 
# Fits an SVM to training_samples produced by text_encoding file 
# 

import joblib, numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, GridSearchCV 
from scipy.stats import loguniform 
from sklearn.calibration import CalibratedClassifierCV
from text_encoding import load_samples


SEED = np.random.randint(0, high=np.iinfo(np.int32).max)

def get_linear_svc(X, y): 
    '''
    Yields the best performing model (in terms of hyperparameters) given training set

    Inputs: 
        X - training_samples 
        y - training_labels 
    Output: 
        Best performing model from randomized search, then a finer grid search
    '''
    svc = LinearSVC(class_weight="balanced", dual=False, loss="squared_hinge", penalty="l2")
    param_grid = {
        "C": loguniform(1e-3, 1e+1),
        "tol": [1e-4, 1e-3],
    }

    cross = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    search = RandomizedSearchCV(svc, param_distributions=param_grid, n_iter=30, scoring="average_precision", 
                                cv=cross, n_jobs=-1, random_state=SEED, verbose=51, refit=True)
    search.fit(X, y)
    C, tol = search.best_params_["C"], search.best_params_["tol"]

    best_grid = {
        "C": np.logspace(np.log10(C/5), np.log10(C * 5), 9),
        "tol": np.logspace(np.log10(tol/5), np.log10(tol * 5), 9)
    }

    fine = GridSearchCV(svc, param_grid=best_grid, cv=cross, 
                        n_jobs=-1, verbose=51, refit=True)
    fine.fit(X, y)
    return fine.best_estimator_


def get_rbf_svc(X, y):
    '''
    Identical to get_linear_svc but for rbf kernel hyperparams 

    Inputs: 
        X - training_samples 
        y - training_labels 
    Output: 
        Best performing model from randomized search, then a finer grid search
    '''
    cross = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    svc = SVC(kernel="rbf", class_weight="balanced", probability=False)

    param_grid = {
        "C": loguniform(1e-3, 10),
        "gamma": loguniform(1e-5, 1)
    }

    search = RandomizedSearchCV(svc, param_distributions=param_grid, n_iter=120, scoring="average_precision",
                                cv=cross, n_jobs=-1, random_state=SEED, verbose=51, refit=True)
    search.fit(X, y)
    C, gamma = search.best_params_["C"], search.best_params_["gamma"]

    best_grid = { 
        "C": np.logspace(np.log10(C/5), np.log10(C * 5), 9),
        "gamma": np.logspace(np.log10(gamma/5), np.log10(gamma * 5), 9)
    }

    fine = GridSearchCV(svc, param_grid=best_grid, cv=cross, 
                        n_jobs=-1, verbose=51, refit=True)
    fine.fit(X, y)
    return fine.best_estimator_


def main():
    # Load training_samples 
    X, y = load_samples("training_samples.npz")

    rbf = get_rbf_svc(X, y)

    calibrated = CalibratedClassifierCV(rbf, method="sigmoid", cv=5)
    calibrated.fit(X, y)

    joblib.dump(calibrated, "svm_best.joblib")


if __name__ == "__main__":
    main()

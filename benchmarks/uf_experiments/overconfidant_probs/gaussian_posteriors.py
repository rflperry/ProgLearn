#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from proglearn.forest import UncertaintyForest
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import defaultdict
import pickle
from datasets import (
    sample_2gaussians,
    sample_2gaussians_oblique,
    sample_3gaussians,
    sample_3gaussians_oblique,
    sample_trunk
)

# %%
n = 100
d = 20
var = 1
n_runs = 5
n_estimators = 100
kappas = [0.01, 0.1, 0.5, 1, 5]
epsilons = [1]# np.linspace(0, 2, 9)
n_jobs = 40

n_gaussians = 2
tag = 'oblique'

clfs = [
    (
        f"UF (k={k})",
        UncertaintyForest(
            n_estimators=n_estimators, tree_construction_proportion=0.4, kappa=k
        )
    )
    for k in kappas
]

clfs += [
    ('IRF', CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs),
            method="isotonic",
            cv=5,
        )),
    ('SigRF', CalibratedClassifierCV(
            base_estimator=RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs),
            method="sigmoid",
            cv=5,
        )),
    ('UF (Uncorrected)', UncertaintyForest(
            n_estimators=n_estimators, tree_construction_proportion=0.4
        )),
    ('RF', RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs))
]

for name, clf in clfs:
    results = {
        "algo": name,
        "epsilons": epsilons,
        "y": [],
        "y_probs": [],
        "y_probs_analytic": []
    }
    for epsilon in epsilons:
        y = []
        y_probs = []
        y_probs_analytic = []
        for run in range(n_runs):
            if tag == 'oblique' and n_gaussians == 2:
                sample_func = sample_2gaussians_oblique
            elif tag == '' and n_gaussians == 2:
                sample_func = sample_2gaussians
            elif tag == 'trunk' and n_gaussians == 2:
                sample_trunk
            # elif oblique and n_gaussians == 3:
            #     sample_func = sample_3gaussians_oblique
            # elif n_gaussians == 3:
            #     sample_func = sample_3gaussians

            X_train, y_train, probs = sample_func(
                n, epsilon, d, var, calculate_prob=True
            )
            X_test, y_test, probs = sample_func(n, epsilon, d, var, calculate_prob=True)

            clf.fit(X_train, y_train)
            y_hat_probs = clf.predict_proba(X_test)

            y.append(y_test)
            y_probs.append(y_hat_probs)
            y_probs_analytic = [probs]

        results['y'].append(y)
        results['y_probs'].append(y_probs)
        results['y_probs_analytic'].append(y_probs_analytic)
            
    if tag != '':
        tag = '_' + tag
    with open(
        f"./results_vs_epsilon/{name}_{n}x{d}{tag}.pickle", "wb"
    ) as f:
        pickle.dump(results, f)

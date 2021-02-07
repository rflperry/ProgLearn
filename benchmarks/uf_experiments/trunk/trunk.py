#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from proglearn.forest import UncertaintyForest
from joblib import Parallel, delayed
from tqdm import tqdm
from collections import defaultdict
import pickle
from datasets import sample_trunk

# %%
def unpickle(file):
    with open(file, "rb") as f:
        data = pickle.load(f, encoding="bytes")
    return data


ns = (np.asarray([10, 100, 1000, 10000]) / 2).astype(int)  # 2 classes
d = 20
var = 1
n_runs = 20
n_estimators = 100
kappas = [0.01, 0.1, 1, 10]
epsilon = 1
n_jobs = 40

tag = "trunk"
if tag != "" and tag[0] != "_":
    tag = "_" + tag

clfs = [
    (
        f"UF (0.5-split, k={k})",
        UncertaintyForest(
            n_estimators=n_estimators, tree_construction_proportion=0.5, kappa=k
        )
    )
    for k in kappas
]

clfs += [
    # ('IRF', CalibratedClassifierCV(
    #         base_estimator=RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs),
    #         method="isotonic",
    #         cv=5,
    #     )),
    # ('SigRF', CalibratedClassifierCV(
    #         base_estimator=RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs),
    #         method="sigmoid",
    #         cv=5,
    #     )),
    ('UF (0.5-split, Uncorrected)', UncertaintyForest(
            n_estimators=n_estimators, tree_construction_proportion=0.5
        )),
    # ("RF", RandomForestClassifier(n_estimators=n_estimators, n_jobs=n_jobs))
]


for name, clf in clfs:
    results = {
        "algo": name,
        "sample_sizes": ns,
        "y": [],
        "y_probs": [],
        "y_probs_analytic": [],
    }
    fname = f"./results_vs_n/{name}_{epsilon}-{d}{tag}.pickle"
    for n in ns:
        y = []
        y_probs = []
        y_probs_analytic = []
        for run in range(n_runs):
            sample_func = sample_trunk

            X_train, y_train = sample_func(n, epsilon, d, var, calculate_prob=False)
            X_test, y_test, probs = sample_func(n, epsilon, d, var, calculate_prob=True)

            clf.fit(X_train, y_train)
            y_hat_probs = clf.predict_proba(X_test)

            y.append(y_test)
            y_probs.append(y_hat_probs)
            y_probs_analytic.append(probs)

        results["y"].append(y)
        results["y_probs"].append(y_probs)
        results["y_probs_analytic"].append(y_probs_analytic)

        try:  # append to previous file
            old = unpickle(fname)
            keys = ["y", "y_probs", "y_probs_analytic"]
            if n in old["sample_sizes"]:
                i = np.where(old["sample_sizes"] == n)[0][0]
                for key in keys:
                    results[key][i] = old[key][i] + results[key][-1]
            else:
                results["sample_size"] += n
                for key in keys:
                    results[key] = old[key] + results[key]
        except:  # make a new one
            pass

    with open(fname, "wb") as f:
        pickle.dump(results, f)

#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from proglearn.forest import UncertaintyForest
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle
from datasets import sample_2gaussians_oblique, sample_2gaussians, sample_3gaussians, sample_3gaussians_oblique

#%%
# import matplotlib.pyplot as plt
# X, y = sample_2gaussians(1000, epsilon=1)
# plt.scatter(X[:,0], X[:,1], c=y)
# plt.show()
# %%
n = 1000
d = 200
var = 1
n_runs = 5
n_estimators = 100
epsilons = np.linspace(0, 2, 11)
n_jobs = 40


clfs = [
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

n_gaussians = 3
oblique = True


for name, clf in clfs:
    for epsilon in epsilons:
        for run in range(n_runs):
            if oblique and n_gaussians == 2:
                sample_func = sample_2gaussians_oblique
            elif n_gaussians == 2:
                sample_func = sample_2gaussians
            elif oblique and n_gaussians == 3:
                sample_func = sample_3gaussians_oblique
            elif n_gaussians == 3:
                sample_func = sample_3gaussians

            X_train, y_train = sample_func(n, epsilon, d, var)
            X_test, y_test = sample_func(n, epsilon, d, var)

            clf.fit(X_train, y_train)
            y_probs = clf.predict_proba(X_test)

            results = {
                'algo': name,
                'epsilon': epsilon,
                'y_probs': y_probs,
                'y': y_test,
            }
            if oblique:
                tag = '_oblique'
            else:
                tag = ''
            with open(f'./{n_gaussians}gaussians{tag}_results/{name}_{n}x{d}_{var}_{epsilon:.2f}_{run}.pickle', 'wb') as f:
                pickle.dump(results, f)

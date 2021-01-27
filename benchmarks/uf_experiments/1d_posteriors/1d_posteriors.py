#%%
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from proglearn.forest import UncertaintyForest
from joblib import Parallel, delayed
from tqdm import tqdm
import pickle

def generate_data(n, mean, var):
    """
    Parameters
    ---
    n : int
        The number of data to be generated
    mean : double
        The mean of the data to be generated
    var : double
        The variance in the data to be generated
    """
    y = 2 * np.random.binomial(1, 0.5, n) - 1  # classes are -1 and 1.
    # X = np.random.multivariate_normal(
    #     mean * y, var * np.eye(n), 1
    # ).T  # creating the X values using
    X = np.random.normal(mean * y, var, (1, 6000)).T
    # the randomly distributed y that were generated in the line above

    return X, y


def estimate_posterior(algo, n, mean, var, num_trials, X_eval, parallel=False):
    """
    Estimate posteriors for many trials and evaluate in the given X_eval range

    Parameters
    ---
    algo : dict
        A dictionary of the learner to be used containing a key "instance" of the learner
    n : int
        The number of data to be generated
    mean : double
        The mean of the data used
    var : double
        The variance of the data used
    num_trials : int
        The number of trials to run over
    X_eval : list
        The range over which to evaluate X values for
    """
    obj = algo["instance"]  # grabbing the instance of the learner

    def worker(t):
        X, y = generate_data(n, mean, var)  # generating data with the function above
        obj.fit(X, y)  # using the fit function of the learner to fit the data

        # using the predict_proba function on the range of desired X
        return obj.predict_proba(X_eval)[:, 1]

    if parallel:
        predicted_posterior = np.array(
            Parallel(n_jobs=-2)(delayed(worker)(t) for t in range(num_trials))
        )
    else:
        predicted_posterior = np.zeros((num_trials, X_eval.shape[0]))
        for t in tqdm(range(num_trials)):
            predicted_posterior[t, :] = worker(t)

    return predicted_posterior


#%%
# Here are the "Real Parameters"
n = 6000  # number of data points
mean = 1  # mean of the data
var = 1  # variance of the data
num_trials = 100  # number of trials to run
X_eval = np.linspace(-2, 2, num=30).reshape(
    -1, 1
)  # the evaluation span (over X) for the plot
n_estimators = 300  # the number of estimators
num_plotted_trials = 10  # the number of "fainter" lines to be displayed on the figure


# Algorithms used to produce figure 1
algos = [
    # {
    #     "instance": RandomForestClassifier(n_estimators=n_estimators),
    #     "label": "CART",
    #     "title": "CART Forest",
    #     "color": "#1b9e77",
    # },
    # {
    #     "instance": CalibratedClassifierCV(
    #         base_estimator=RandomForestClassifier(n_estimators=n_estimators // 5),
    #         method="isotonic",
    #         cv=5,
    #     ),
    #     "label": "IRF",
    #     "title": "Isotonic Reg. Forest",
    #     "color": "#fdae61",
    # },
    {
        "instance": UncertaintyForest(
            n_estimators=n_estimators, tree_construction_proportion=0.4, kappa=3.0
        ),
        "label": "UF",
        "title": "Uncertainty Forest",
        "color": "#F41711",
    },
]

# Plotting parameters
parallel = False


#%%
# This is the code that actually generates data and predictions.
for algo in algos:
    algo["predicted_posterior"] = estimate_posterior(
        algo, n, mean, var, num_trials, X_eval, parallel=parallel
    )
    with open(f'./results/{algo["label"]}.pickle', 'wb') as f:
        pickle.dump(algo, f)
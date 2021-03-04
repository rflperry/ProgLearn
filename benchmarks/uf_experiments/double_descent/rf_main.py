import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from tqdm import tqdm

from proglearn import UncertaintyForest

from sklearn.metrics import log_loss, brier_score_loss, accuracy_score
from joblib import Parallel, delayed
import multiprocessing

import argparse

from utils import gini_impurity_mean, hellinger_explicit, pdf, generate_gaussian_parity, predict_insample


def get_tree(method="rf", max_depth=1, n_estimators=1, max_leaf_nodes=None):
    if method == "gb":
        rf = GradientBoostingClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=1514,
            max_leaf_nodes=max_leaf_nodes,
            learning_rate=1,
            criterion="mse",
        )
    elif method == "rf":
        rf = RandomForestClassifier(
            bootstrap=False,
            max_depth=max_depth,
            n_estimators=n_estimators,
            random_state=1514,
            max_leaf_nodes=max_leaf_nodes,
        )
    elif method == "uf":
        rf = UncertaintyForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            kappa=1,
            tree_construction_proportion=0.5,
        )
    else:
        raise ValueError(f"method {method} not a valid input")

    return rf


def rf_dd_exp(
    N=4096, reps=100, max_node=None, n_est=10, exp_alias="depth", method="rf"
):

    xx, yy = np.meshgrid(np.arange(-2, 2, 4 / 100), np.arange(-2, 2, 4 / 100))
    true_posterior = np.array([pdf(x) for x in (np.c_[xx.ravel(), yy.ravel()])])

    train_mean_error, test_mean_error = [], []
    train_mean_error_log, test_mean_error_log = [], []
    gini_train_mean_score, gini_test_mean_score = [], []

    # X, y = get_sample(N)
    X_train, y_train = generate_gaussian_parity(n=N, angle_params=0)
    X_test, y_test = generate_gaussian_parity(n=N, angle_params=0)

    if max_node is None:
        rf = get_tree(method, max_depth=None)
        rf.fit(X_train, y_train)
        if method == "gb":
            max_node = (
                sum([estimator[0].get_n_leaves() for estimator in rf.estimators_])
            ) + 50
        elif method == "rf":
            max_node = (
                sum([estimator.get_n_leaves() for estimator in rf.estimators_]) + 50
            )
        elif method == "uf":
            max_node = (
                sum(
                    [
                        estimator.transformer_.get_n_leaves()
                        for estimator in rf.lf_.pl_.transformer_id_to_transformers[0]
                    ]
                )
                + 50
            )
        else:
            raise ValueError(f"method {method} not a valid input")

    train_error, test_error = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    train_transformers_error = [list() for _ in range(reps)]
    train_voters_error = [list() for _ in range(reps)]

    train_error_log, test_error_log = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    train_transformers_error_log = [list() for _ in range(reps)]
    train_voters_error_log = [list() for _ in range(reps)]

    gini_score_train, gini_score_test = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    gini_score_train_transformers = [list() for _ in range(reps)]
    gini_score_train_voters = [list() for _ in range(reps)]

    ece_error = [list() for _ in range(reps)]
    nodes = [list() for _ in range(reps)]
    polys = [list() for _ in range(reps)]
    # for depth in tqdm(range(1, max_node + n_est), position=0, leave=True):
    # for rep_i in tqdm(range(reps), position=0, leave=True):
    def one_run(rep_i):
        print(rep_i)
        X_train, y_train = generate_gaussian_parity(n=N, angle_params=0)
        X_test, y_test = generate_gaussian_parity(n=1000, angle_params=0)

        rf = get_tree(method, max_depth=1)
        # for rep_i in range(reps):
        for depth in tqdm(range(1, max_node + n_est), position=0, leave=True):

            if depth < max_node:
                rf.max_depth += 1
            else:
                rf.n_estimators += 3
                rf.max_depth += 15
                # rf.warm_start=True

            rf.fit(X_train, y_train)

            if method == "gb":
                nodes[rep_i].append(
                    sum([(estimator[0].get_n_leaves()) for estimator in rf.estimators_])
                )
            elif method == "rf":
                nodes[rep_i].append(
                    sum([estimator.get_n_leaves() for estimator in rf.estimators_])
                )
            elif method == "uf":
                # UF -> LLClassificationForest -> ClassificationProgLearner -> TreeClassificationTransformer -> DecisionTree
                nodes[rep_i].append(
                    sum(
                        [
                            estimator.transformer_.get_n_leaves()
                            for estimator in rf.lf_.pl_.transformer_id_to_transformers[0]
                        ]
                    )
                )
            else:
                raise ValueError(f"method {method} not a valid input")

            if method == "uf":
                leaf_idxs = np.asarray(
                    [
                        estimator.transformer_.apply(X_train)
                        for estimator in rf.lf_.pl_.transformer_id_to_transformers[0]
                    ]
                ).T
                train_voters_error[rep_i].append(
                    1 - accuracy_score(y_train, predict_insample(rf, method, X_train, 'voters')))
                train_transformers_error[rep_i].append(
                    1 - accuracy_score(y_train, predict_insample(rf, method, X_train, 'transformers')))
                gini_score_train_voters[rep_i].append(gini_impurity_mean(rf, X_train, y_train, 'voters'))
                gini_score_train_transformers[rep_i].append(gini_impurity_mean(rf, X_train, y_train, 'transformers'))
                train_voters_error_log[rep_i].append(
                    log_loss(y_train, predict_insample(rf, method, X_train, 'voters', return_proba=True)))
                train_transformers_error_log[rep_i].append(
                    log_loss(y_train, predict_insample(rf, method, X_train, 'transformers', return_proba=True)))
            else:
                leaf_idxs = rf.apply(X_train)

            train_error[rep_i].append(1 - accuracy_score(y_train, rf.predict(X_train)))
            gini_score_train[rep_i].append(gini_impurity_mean(rf, X_train, y_train))
            train_error_log[rep_i].append(log_loss(y_train, rf.predict_proba(X_train)))
    
            test_error[rep_i].append(1 - accuracy_score(y_test, rf.predict(X_test)))
            gini_score_test[rep_i].append(gini_impurity_mean(rf, X_test, y_test))
            test_error_log[rep_i].append(log_loss(y_test, rf.predict_proba(X_test)))

            polys[rep_i].append(len(np.unique(leaf_idxs)))
            
            #             ece_error[rep_i].append( brier_score_loss(y_train, rf.predict(X_train)))
            rf_posteriors_grid = rf.predict_proba(np.c_[xx.ravel(), yy.ravel()])
            ece_error[rep_i].append(
                hellinger_explicit(rf_posteriors_grid, true_posterior)
            )
        # nodes[rep_i] = np.array(nodes[rep_i])
        # polys[rep_i] = np.array(polys[rep_i])
        # train_error[rep_i] = np.array(train_error[rep_i])
        # test_error[rep_i] = np.array(test_error[rep_i])
        # train_error_log[rep_i] = np.array(train_error_log[rep_i])
        # test_error_log[rep_i] = np.array(test_error_log[rep_i])
        # gini_score_train[rep_i] = np.array(gini_score_train[rep_i])
        # gini_score_test[rep_i] = np.array(gini_score_test[rep_i])
        # ece_error[rep_i] = np.array(ece_error[rep_i])

        np.save(
            f"results/xor_{method}_dd_" + exp_alias + "_" + str(rep_i) + ".npy",
            [
                nodes[rep_i],
                polys[rep_i],
                train_error[rep_i],
                train_transformers_error[rep_i],
                train_voters_error[rep_i],
                test_error[rep_i],
                train_error_log[rep_i],
                train_transformers_error_log[rep_i],
                train_voters_error_log[rep_i],
                test_error_log[rep_i],
                gini_score_train[rep_i],
                gini_score_train_transformers[rep_i],
                gini_score_train_voters[rep_i],
                gini_score_test[rep_i],
                ece_error[rep_i],
            ],
        )

    num_cores = multiprocessing.cpu_count()

    Parallel(n_jobs=-1)(delayed(one_run)(i) for i in range(reps))

    train_mean_error = np.array(train_error).mean(axis=0)
    test_mean_error = np.array(test_error).mean(axis=0)
    train_mean_error_log = np.array(train_error_log).mean(axis=0)
    test_mean_error_log = np.array(test_error_log).mean(axis=0)
    nodes_mean = np.array(nodes).mean(axis=0)
    gini_train_mean_score = np.array(gini_score_train).mean(axis=0)
    gini_test_mean_score = np.array(gini_score_test).mean(axis=0)
    error_dict = {
        "train_err": train_mean_error,
        "test_err": test_mean_error,
        "train_err_log": train_mean_error_log,
        "test_err_log": test_mean_error_log,
        "train_gini": gini_train_mean_score,
        "test_gini": gini_test_mean_score,
        "nodes": nodes_mean,
    }
    return error_dict


parser = argparse.ArgumentParser(description="Run a double descent experiment.")

parser.add_argument("--depth", action="store_true", default=False)
parser.add_argument("--reps", action="store", dest="reps", type=int, default=100)
parser.add_argument("--n_est", action="store", dest="n_est", type=int, default=10)
parser.add_argument(
    "--max_node", action="store", dest="max_node", default=None, type=int
)
parser.add_argument(
    "--cov_scale", action="store", dest="cov_scale", default=1.0, type=float
)
parser.add_argument(
    "--method", action="store", dest="method", default="rf", choices=["rf", "gb", "uf"]
)

args = parser.parse_args()

exp_alias = "depth" if args.depth else "width"

error_dd = rf_dd_exp(
    max_node=args.max_node,
    n_est=args.n_est,
    reps=args.reps,
    exp_alias=exp_alias,
    method=args.method,
)

import numpy as np


def generate_gaussian_parity(n, cov_scale=1, angle_params=None, k=1, acorn=None):
    #     means = [[-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [-1.5, 1.5]]
    means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    blob = np.concatenate(
        [
            np.random.multivariate_normal(
                mean, cov_scale * np.eye(len(mean)), size=int(n / 4)
            )
            for mean in means
        ]
    )

    X = np.zeros_like(blob)
    Y = np.concatenate([np.ones((int(n / 4))) * int(i < 2) for i in range(len(means))])
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(
        angle_params * np.pi / 180
    )
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(
        angle_params * np.pi / 180
    )
    return X, Y.astype(int)


def gini_impurity(P1=0, P2=0):
    denom = P1 + P2
    Ginx = 2 * (P1 / denom) * (P2 / denom)
    if Ginx == 0: # correction for honest forests
        return 0
    return Ginx


def gini_impurity_mean(rf, data, label, subset='all'):
    """    
    subset : choice of insample subset for honest forest
        - 'all' : uses all
        - 'voters' : predict on just voter data
        - 'transformers' : predict on just transformer data
    """
    label_subsets = []
    predict = None
    try:
        assert subset == 'all'
        predict = label
        leaf_idxs = rf.apply(data).T
    except:  # In case UF
        X_idx = np.arange(data.shape[0])
        leaf_idxs = []
        for i, tree in enumerate(rf.lf_.pl_.transformer_id_to_transformers[0]):
            subset_idx = rf.lf_.pl_.task_id_to_bag_id_to_voter_data_idx[0][i]
            if subset == 'transformers':
                # Predict on transformer data
                subset_idx = np.delete(X_idx, subset_idx)
            elif subset == 'all':
                subset_idx = X_idx
            label_subsets.append(label[subset_idx])
            leaf_idxs.append(tree.transformer_.apply(data[subset_idx]))
    
    gini_mean_score = []
    for t in range(len(leaf_idxs)):
        gini_arr = []
        for l in np.unique(leaf_idxs[t]):
            if predict is not None:
                cur_l_idx = predict[leaf_idxs[t] == l]
            else:
                cur_l_idx = label_subsets[t][leaf_idxs[t] == l]
            pos_count = np.sum(cur_l_idx)
            neg_count = len(cur_l_idx) - pos_count
            gini = gini_impurity(pos_count, neg_count)
            gini_arr.append(gini)

        gini_mean_score.append(np.array(gini_arr).mean())
    return np.array(gini_mean_score).mean()


def predict_insample(rf, method, X, subset='all', return_proba=False):
    """
    For honest forest, in-sample has voter and transformer subsets
    
    subset : choice of insample subset
        - 'all' : predict on X
        - 'voters' : predict on just voter data
        - 'transformers' : predict on just transformer data
    """
    if subset == 'all' or method != 'uf':
        proba = rf.predict_proba(X)
        if return_proba:
            return proba
        else:
            return np.argmax(proba, 1)
    X_idx = np.arange(X.shape[0])
    X_counts = np.zeros((X.shape[0], 1))
    X_proba = None
    for i, tree in enumerate(rf.lf_.pl_.transformer_id_to_transformers[0]):
        subset_idx = rf.lf_.pl_.task_id_to_bag_id_to_voter_data_idx[0][i]
        if subset == 'transformers':
            # Predict on transformer data
            subset_idx = np.delete(X_idx, subset_idx)
        # Keep track of counts for averaging at the end
        X_counts[subset_idx] += 1

        # Get subset probabilities
        proba = rf.predict_proba(X[subset_idx])
        if X_proba is None:
            X_proba = np.zeros((X.shape[0], proba.shape[1]))
        X_proba[subset_idx] += proba

    # Average, w/ finite sample correction
    unsampled_idx = np.where(X_counts == 0)[0]
    X_proba[unsampled_idx] += 0.5
    X_counts[unsampled_idx] = 1
    X_proba /= X_counts

    if return_proba:
        return X_proba
    else:
        return np.argmax(X_proba, 1)


def hellinger_explicit(p, q):
    """Hellinger distance between two discrete distributions.
    Same as original version but without list comprehension
    """
    return np.mean(np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=1)) / np.sqrt(2))


def pdf(x):
    mu01, mu02, mu11, mu12 = [[-1, -1], [1, 1], [-1, 1], [1, -1]]

    cov = 1 * np.eye(2)
    inv_cov = np.linalg.inv(cov)

    p0 = (
        np.exp(-(x - mu01) @ inv_cov @ (x - mu01).T)
        + np.exp(-(x - mu02) @ inv_cov @ (x - mu02).T)
    ) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

    p1 = (
        np.exp(-(x - mu11) @ inv_cov @ (x - mu11).T)
        + np.exp(-(x - mu12) @ inv_cov @ (x - mu12).T)
    ) / (2 * np.pi * np.sqrt(np.linalg.det(cov)))

    return [p1 / (p0 + p1), p0 / (p0 + p1)]


def read_results(reps, exp_alias="depth", method="rf"):
    train_error, test_error = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    train_error_log, test_error_log = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    gini_score_train, gini_score_test = [list() for _ in range(reps)], [
        list() for _ in range(reps)
    ]
    nodes = [list() for _ in range(reps)]
    polys = [list() for _ in range(reps)]
    ece_error = [list() for _ in range(reps)]

    for rep_i in range(reps):
        [
            nodes[rep_i],
            polys[rep_i],
            train_error[rep_i],
            test_error[rep_i],
            train_error_log[rep_i],
            test_error_log[rep_i],
            gini_score_train[rep_i],
            gini_score_test[rep_i],
            ece_error[rep_i],
        ] = np.load(f"results/xor_{method}_dd_" + exp_alias + "_" + str(rep_i) + ".npy")

    train_mean_error = np.array(train_error).mean(axis=0)
    test_mean_error = np.array(test_error).mean(axis=0)
    train_mean_error_log = np.array(train_error_log).mean(axis=0)
    test_mean_error_log = np.array(test_error_log).mean(axis=0)
    nodes_mean = np.array(nodes).mean(axis=0)
    polys_mean = np.array(polys).mean(axis=0)
    gini_train_mean_score = np.array(gini_score_train).mean(axis=0)
    gini_test_mean_score = np.array(gini_score_test).mean(axis=0)
    ece_mean_error = np.array(ece_error).mean(axis=0)
    error_dict = {
        "train_err": train_mean_error,
        "test_err": test_mean_error,
        "train_err_log": train_mean_error_log,
        "test_err_log": test_mean_error_log,
        "train_gini": gini_train_mean_score,
        "test_gini": gini_test_mean_score,
        " ece_error": ece_mean_error,
        "polys": polys_mean,
        "nodes": nodes_mean,
    }

    return error_dict

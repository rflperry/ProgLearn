import numpy as np
from scipy.stats import norm


def sample_trunk(n, epsilon=1, d=2, var=1, calculate_prob=False):
    """
    Parameters
    ---
    n : int
        The number of data to be generated from each gaussian
    mean : double
        The mean of the data to be generated
    var : double
        The variance in the data to be generated
    calculate_prob : bool (default False)
        If True, returns the analytic probabilities of being in each class.
        Note, class priors is uniform.
    """
    mu1 = -epsilon / np.sqrt(np.arange(d) + 1)
    mu2 = -mu1
    # mu1/mu2 distance of epsilon away along the diagonal in all dimensions
    y = np.hstack([i * np.ones(n) for i in range(2)])

    X1 = np.random.normal(mu1, var, (n, d))
    X2 = np.random.normal(mu2, var, (n, d))
    X = np.vstack((X1, X2))

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    if calculate_prob:
        proj_vec = 1 / np.sqrt(np.arange(d) + 1)  # magnitude 1
        proj_vec /= np.linalg.norm(proj_vec)
        class_probs = np.zeros((X.shape[0], 2))
        mean_norm = np.linalg.norm(
            mu1
        )  # sign/direction invariant, to add in pdf calculation
        # Scaled by class prior
        class_probs[:, 0] = (
            norm.pdf(np.dot(X, proj_vec), loc=-mean_norm, scale=np.sqrt(var)) * 0.5
        )
        class_probs[:, 1] = (
            norm.pdf(np.dot(X, proj_vec), loc=mean_norm, scale=np.sqrt(var)) * 0.5
        )
        # normalize
        class_probs /= class_probs.sum(axis=1, keepdims=True)

        return X, y, class_probs
    else:
        return X, y

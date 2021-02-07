import numpy as np
from scipy.stats import norm

def sample_2gaussians(n, epsilon, d=2, var=1, calculate_prob=False):
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
    mu1 = np.zeros(d)
    mu2 = np.zeros(d)
    mu2[0] = epsilon
    y = np.hstack([i*np.ones(n) for i in range(2)])

    X1 = np.random.normal(mu1, np.sqrt(var), (n, d))
    X2 = np.random.normal(mu2, np.sqrt(var), (n, d))
    X = np.vstack((X1, X2))

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]
    if calculate_prob:
        class_probs = np.zeros((X.shape[0], 2))
        # Scaled by class prior
        class_probs[:, 0] = norm.pdf(X[:, 0], loc=0, scale=np.sqrt(var)) * 0.5
        class_probs[:, 1] = norm.pdf(X[:, 0], loc=epsilon, scale=np.sqrt(var)) * 0.5
        # normalize
        class_probs /= class_probs.sum(axis=1, keepdims=True)

        return X, y, class_probs
    else:
        return X, y


def sample_2gaussians_oblique(n, epsilon, d=2, var=1, calculate_prob=False):
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
    mu1 = np.zeros(d)
    mu2 = np.zeros(d)
    # mu2 is distance of epsilon away along the diagonal in all dimensions
    mu2 += epsilon / np.sqrt(d)
    y = np.hstack([i*np.ones(n) for i in range(2)])

    X1 = np.random.normal(mu1, var, (n, d))
    X2 = np.random.normal(mu2, var, (n, d))
    X = np.vstack((X1, X2))

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    if calculate_prob:
        proj_vec = np.ones(d) / np.sqrt(d) # magnitude 1
        class_probs = np.zeros((X.shape[0], 2))
        # Scaled by class prior
        class_probs[:, 0] = norm.pdf(np.dot(X, proj_vec), loc=0, scale=np.sqrt(var)) * 0.5
        class_probs[:, 1] = norm.pdf(np.dot(X, proj_vec), loc=epsilon, scale=np.sqrt(var)) * 0.5
        # normalize
        class_probs /= class_probs.sum(axis=1, keepdims=True)

        return X, y, class_probs
    else:
        return X, y


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
    mu2 = epsilon / np.sqrt(np.arange(d) + 1)
    # mu2 is distance of epsilon away along the diagonal in all dimensions
    y = np.hstack([i*np.ones(n) for i in range(2)])

    X1 = np.random.normal(mu1, var, (n, d))
    X2 = np.random.normal(mu2, var, (n, d))
    X = np.vstack((X1, X2))

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    if calculate_prob:
        proj_vec = 1 / np.sqrt(np.arange(d) + 1) # magnitude 1
        proj_vec /= np.linalg.norm(proj_vec)
        class_probs = np.zeros((X.shape[0], 2))
        # Scaled by class prior
        class_probs[:, 0] = norm.pdf(np.dot(X, proj_vec), loc=np.linalg.norm(mu1), scale=np.sqrt(var)) * 0.5
        class_probs[:, 1] = norm.pdf(np.dot(X, proj_vec), loc=np.linalg.norm(mu2), scale=np.sqrt(var)) * 0.5
        # normalize
        class_probs /= class_probs.sum(axis=1, keepdims=True)

        return X, y, class_probs
    else:
        return X, y


def sample_3gaussians(n, epsilon, d=2, var=1):
    """
    Parameters
    ---
    n : int
        The number of data to be generated from each gaussian
    mean : double
        The mean of the data to be generated
    var : double
        The variance in the data to be generated
    """
    mu1 = np.zeros(d)
    mu2 = np.zeros(d)
    mu3 = np.zeros(d)
    mu2[0] = epsilon
    # mu3[:2] = [epsilon / 2, epsilon * np.sqrt(3) / 2]
    mu3[1] = epsilon

    y = np.hstack([i*np.ones(n) for i in range(3)])
    X = np.vstack([
        np.random.normal(m, var, (n, d)) for m in (mu1, mu2, mu3)
        ])

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    return X, y

def sample_3gaussians_oblique(n, epsilon, d=2, var=1):
    """
    Parameters
    ---
    n : int
        The number of data to be generated from each gaussian
    mean : double
        The mean of the data to be generated
    var : double
        The variance in the data to be generated
    """
    mu1 = np.zeros(d)
    mu2 = np.zeros(d)
    mu3 = np.zeros(d)
    # mu2 is distance of epsilon away along the diagonal in all dimensions
    mu2 += epsilon / np.sqrt(d)
    mu3 += epsilon / np.sqrt(d)
    mu3 *= np.asarray([-1, 1]*(len(mu3) // 2))
    y = np.hstack([i*np.ones(n) for i in range(3)])

    X1 = np.random.normal(mu1, var, (n, d))
    X2 = np.random.normal(mu2, var, (n, d))
    X = np.vstack((X1, X2))

    y = np.hstack([i*np.ones(n) for i in range(3)])
    X = np.vstack([
        np.random.normal(m, var, (n, d)) for m in (mu1, mu2, mu3)
        ])

    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)

    X = X[idx]
    y = y[idx]

    return X, y
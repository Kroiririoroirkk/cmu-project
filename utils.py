import numpy as np


def highest_eig(M):
    """Find the magnitude of the largest eigenvalue of a matrix.

    Parameters
    ----------
    M : np.ndarray, shape (n, n)
        A square matrix

    Returns
    -------
    lam : float
        The magnitude of the largest eigenvalue (in magnitude)
    """
    return np.max(np.abs(np.linalg.eig(M).eigenvalues))


def mk_rand_matrix(rng, n):
    """Generate a Gaussian random square matrix.

    Parameters
    ----------
    rng : np.random.Generator
        The RNG to use
    n : int
        The number of columns of the matrix

    Returns
    -------
    M : np.ndarray, shape (n, n)
        The random matrix
    """
    return rng.multivariate_normal(np.zeros(n),np.eye(n),(n,))


def mk_rand_matrix_envals(rng, envals):
    """Generate a random square matrix with the given eigenvalues.

    Parameters
    ----------
    rng : np.random.Generator
        The RNG to use
    envals : np.ndarray, shape (n,)
        The eigenvalues of the matrix

    Returns
    -------
    M : np.ndarray, shape (n, n)
        The random matrix

    Raises
    ------
    ValueError
        If a NumPy array argument is not of the correct shape.
    """
    n = envals.shape[0]
    
    try:
        assert envals.shape == (n,)
    except AssertionError:
        raise ValueError('Argument not of the correct shape')

    D = np.diag(envals)
    P = mk_rand_matrix(rng, n)

    try:
        return P @ D @ np.linalg.inv(P)
    except np.linalg.LinAlgError:
        return mk_rand_matrix_envals(rng, n, envals)

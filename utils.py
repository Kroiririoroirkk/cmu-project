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

def mk_rand_matrix_3cycl(rng, n, rho=1):
    """Make a random matrix with cyclic (order 3) correlations.

    Parameters
    ----------
    rng : np.random.Generator
        The RNG to use
    n : int
        The number of columns of the matrix
    rho : float
        Strength of the cyclic correlations from -1 to 1 (default 1)

    Returns
    -------
    M : np.ndarray, shape (n, n)
        The random matrix
    """
    M = mk_rand_matrix(rng, n)/np.sqrt(n)
    for i in range(n):
        for j in range(n):
            sum_3cycl = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if rng.random() < np.abs(rho):
                    sum_3cycl += M[i,j]*M[j,k]*M[k,i]
            if rho > 0:
                if sum_3cycl < 0:
                    M[i,j] *= -1
            else:
                if sum_3cycl > 0:
                    M[i,j] *= -1
    return M

def mk_rand_matrix_transtrngl(rng, n, rho=1):
    """Make a random matrix with transitive triangle correlations.

    Parameters
    ----------
    rng : np.random.Generator
        The RNG to use
    n : int
        The number of columns of the matrix
    rho : float
        Strength of the transitive triangle correlations from -1 to 1 (default 1)

    Returns
    -------
    M : np.ndarray, shape (n, n)
        The random matrix
    """
    M = mk_rand_matrix(rng, n)/np.sqrt(n)
    for i in range(n):
        for j in range(n):
            sum_transtrngl = 0
            for k in range(n):
                if k == i or k == j:
                    continue
                if rng.random() < np.abs(rho):
                    sum_transtrngl += M[i,j]*M[j,k]*M[i,k]
            if rho > 0:
                if sum_transtrngl < 0:
                    M[i,j] *= -1
            else:
                if sum_transtrngl > 0:
                    M[i,j] *= -1
    return M

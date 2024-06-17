import numpy as np


def calc_loss(xhats, xs, start_from=0):
    """Calculate the loss between the two input arrays.

    More precisely, the loss is defined to be half of the
    sum-of-squares distance between the two arrays.

    Arguments
    ---------
    xhats : np.ndarray, any shape
        The first input array
    xs : np.ndarray, same shape as xhats
        The second input array
    start_from : int
        The time index to start calculating loss from (default 0)

    Returns
    -------
    loss : float
        Half the sum-of-squares distance

    Raises
    ------
    ValueError
        If the two input arrays are not the same shape.
    """
    try:
        assert xhats.shape == xs.shape
    except AssertionError:
        raise ValueError('Shape mismatch')
    return np.sum((xhats[start_from:]-xs[start_from:])**2)/2


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

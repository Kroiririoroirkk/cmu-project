import numpy as np
import scipy
import sklearn


# First, we have to solve the Riccati equation to obtain Sigma_infty
# and K_infty.
def riccati_error(Sigma_infty, n, m, A, Sigma_process, O, Sigma_obs):
    """Find the amount that Sigma_infty deviates from the Riccati equation.

    Given a candidate value for Sigma_infty, calculate the squared
    Frobenius distance between the two sides of the Riccati equation.

    Parameters
    ----------
    Sigma_infty : np.ndarray, shape (n**2,)
        The candidate value, flattened
    n : int
        The dimensionality of the latent state
    m : int
        The dimensionality of the observations
    A : np.ndarray, shape (n, n)
        The transition matrix for the latent state
    Sigma_process : np.ndarray, shape (n, n)
        The covariance of the process noise
    O : np.ndarray, shape (m, n)
        The observation matrix
    Sigma_obs : np.ndarray, shape (m, m)
        The covariance of the observation noise

    Returns
    -------
    d : float
        The squared Frobenius distance

    Raises
    ------
    ValueError
        If a NumPy array argument is not of the correct shape.
    """
    try:
        assert Sigma_infty.shape == (n**2,)
        assert A.shape == (n, n)
        assert Sigma_process.shape == (n, n)
        assert O.shape == (m, n)
        assert Sigma_obs.shape == (m, m)
    except AssertionError:
        raise ValueError('Argument not of the correct shape')
    Sigma_infty = np.reshape(Sigma_infty, (n, n))
    return np.linalg.norm(Sigma_infty -
        (A @ (Sigma_infty - Sigma_infty @ O.T @ np.linalg.inv(
            O @ Sigma_infty @ O.T + Sigma_obs) @ O @ Sigma_infty) @ A.T
        + Sigma_process), ord='fro')**2


def find_steady_state_kalman_parameters(n, m, A, Sigma_process, O, Sigma_obs):
    """Calculate the steady-state Kalman filter parameters.

    Parameters
    ----------
    n : int
        The dimensionality of the latent state
    m : int
        The dimensionality of the observations
    A : np.ndarray, shape (n, n)
        The transition matrix for the latent state
    Sigma_process : np.ndarray, shape (n, n)
        The covariance of the process noise
    O : np.ndarray, shape (m, n)
        The observation matrix
    Sigma_obs : np.ndarray, shape (m, m)
        The covariance of the observation noise

    Returns
    -------
    Sigma_infty : np.ndarray, shape (n, n)
        The steady-state prediction covariance
    K_infty : np.ndarray, shape (n, m)
        The steady-state Kalman gain

    Raises
    ------
    ValueError
        If a NumPy array argument is not of the correct shape.
    sklearn.exceptions.ConvergenceWarning
        If the Riccati equation solver does not converge.
    """
    res = scipy.optimize.minimize(riccati_error, np.eye(n).flatten(), args=(n, m, A, Sigma_process, O, Sigma_obs))
    if not res.success:
        raise sklearn.exceptions.ConvergenceWarning
    Sigma_infty = np.reshape(res.x, (n, n))
    K_infty = Sigma_infty @ np.linalg.inv(Sigma_obs + Sigma_infty)
    return Sigma_infty, K_infty


# Now, we can use the steady-state Kalman filter to infer x-hat values.
def steady_state_kalman_infer(n, m, A, Sigma_process, O, Sigma_obs, ys):
    """Infer hidden Markov latent states using a steady-state Kalman filter.

    Arguments
    ---------
    n : int
        The dimensionality of the latent state
    m : int
        The dimensionality of the observations
    A : np.ndarray, shape (n, n)
        The transition matrix for the latent state
    Sigma_process : np.ndarray, shape (n, n)
        The covariance of the process noise
    O : np.ndarray, shape (m, n)
        The observation matrix
    Sigma_obs : np.ndarray, shape (m, m)
        The covariance of the observation noise
    ys : np.ndarray, shape (num_steps, m)
        The observations

    Returns
    -------
    xhats : np.ndarray, shape (num_steps, n)
        The inferred latent states
    Sigma_infty : np.ndarray, shape (n, n)
        The steady-state prediction covariance
    K_infty : np.ndarray, shape (n, m)
        The steady-state Kalman gain

    Raises
    ------
    sklearn.exceptions.ConvergenceWarning
        If the Riccati equation solver does not converge.
    """
    Sigma_infty, K_infty = find_steady_state_kalman_parameters(n, m, A, Sigma_process, O, Sigma_obs)
    num_steps = ys.shape[0]
    xhats = np.zeros((num_steps, n))
    curr = np.linalg.solve(O, ys[0])
    xhats[0] = curr
    for i in range(1,num_steps):
        curr = A@curr + K_infty@(ys[i] - O@A@curr)
        xhats[i] = curr
    return xhats, Sigma_infty, K_infty

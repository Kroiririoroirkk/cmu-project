import numpy as np
import scipy
import sklearn

from hm_process import HMProcess
from rnn import NeuralNet


# First, we have to solve the Riccati equation to obtain Sigma_infty
# and K_infty.
def riccati_error(Sigma_infty, proc):
    """Find the amount that Sigma_infty deviates from the Riccati equation.

    Given a candidate value for Sigma_infty, calculate the squared
    Frobenius distance between the two sides of the Riccati equation.

    Parameters
    ----------
    Sigma_infty : np.ndarray, shape (n**2,)
        The candidate value, flattened
    proc : HMProcess
        The hidden Markov process of interest

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
        assert Sigma_infty.shape == (proc.n**2,)
    except AssertionError:
        raise ValueError('Argument not of the correct shape')
    Sigma_infty = np.reshape(Sigma_infty, (proc.n, proc.n))
    return np.linalg.norm(Sigma_infty -
        (proc.A @ (Sigma_infty - Sigma_infty @ proc.O.T @ np.linalg.inv(
            proc.O @ Sigma_infty @ proc.O.T + proc.Sigma_obs) @ proc.O @ Sigma_infty) @ proc.A.T
        + proc.Sigma_process), ord='fro')**2


def find_steady_state_kalman_parameters(proc):
    """Calculate the steady-state Kalman filter parameters.

    Parameters
    ----------
    proc : HMProcess
        The hidden Markov process of interest

    Returns
    -------
    Sigma_infty : np.ndarray, shape (n, n)
        The steady-state prediction covariance
    K_infty : np.ndarray, shape (n, m)
        The steady-state Kalman gain
    M_infty : np.ndarray, shape (n, n)
        The steady-state Kalman transition matrix

    Raises
    ------
    ValueError
        If a NumPy array argument is not of the correct shape.
    sklearn.exceptions.ConvergenceWarning
        If the Riccati equation solver does not converge.
    """
    res = scipy.optimize.minimize(riccati_error, np.eye(proc.n).flatten(), args=(proc,))
    if not res.success:
        raise sklearn.exceptions.ConvergenceWarning
    Sigma_infty = np.reshape(res.x, (proc.n, proc.n))
    K_infty = Sigma_infty @ np.linalg.inv(proc.Sigma_obs + Sigma_infty)
    M_infty = proc.A - K_infty@proc.O@proc.A
    return Sigma_infty, K_infty, M_infty


class SteadyStateKalmanFilter:
    """
    A class for representing a Kalman filter whose parameters do not change over time.

    Attributes
    ----------
    process : HMProcess
        The hidden Markov process of interest
    Sigma_infty : np.ndarray, shape (n, n)
        The steady-state prediction covariance
    K_infty : np.ndarray, shape (n, m)
        The steady-state Kalman gain
    M_infty : np.ndarray, shape (n, n)
        The steady-state Kalman transition matrix

    Methods
    -------
    infer(ys)
        Infer hidden Markov latent states using a steady-state Kalman filter.
    """
    
    def __init__(self, proc):
        """
        Arguments
        ---------
        proc : HMProcess
            The hidden Markov process of interest
        
        Raises
        ------
        sklearn.exceptions.ConvergenceWarning
            If the Riccati equation solver does not converge.
        """
        self.process = proc
        self.Sigma_infty, self.K_infty, self.M_infty = find_steady_state_kalman_parameters(proc)
    
    def infer(self, ys):
        """Infer hidden Markov latent states using a steady-state Kalman filter.
    
        Arguments
        ---------
        ys : np.ndarray, shape (num_steps, m)
            The observations
    
        Returns
        -------
        xhats : np.ndarray, shape (num_steps, n)
            The inferred latent states
        """
        nn = NeuralNet(self.M_infty, self.K_infty, np.eye(self.process.n), self.process.x0)
        _, xhats = nn.forward(ys)
        return xhats

    def check_convergence(self):
        """Compare steady-state parameters to actual Kalman filter parameters.

        See https://en.wikipedia.org/wiki/Kalman_filter#Asymptotic_form for
        the equations governing the time evolution of the Kalman filter.
        
        Returns
        -------
        Sigma_infty_dist : np.ndarray, shape (num_steps,)
            Frobenius distance between Sigma_i and Sigma_infty at each time step
        K_infty_dist : np.ndarray, shape (num_steps,)
            Frobenius distance between K_i and K_infty at each time step
        M_infty_dist : np.ndarray, shape (num_steps,)
            Frobenius distance between M_i and M_infty at each time step
        Sigmas : np.ndarray, shape (num_steps, n, n)
            The values of Sigma_i for each i
        Ks : np.ndarray, shape (num_steps, n, m)
            The values of K_i for each i
        Ms : np.ndarray, shape (num_steps, n, n)
            The values of M_i for each i

        Raises
        ------
        LinAlgError
            If inversion of the innovation covariance matrix fails.
        """
        Sigmas = np.zeros((self.process.num_steps, self.process.n, self.process.n)) # Sigmas[i] = Sigma_{i|0...i-1}
        Sigmas_post = np.zeros((self.process.num_steps, self.process.n, self.process.n)) # Sigmas_post[i] = Sigma_{i|0...i}
        Ks = np.zeros((self.process.num_steps, self.process.n, self.process.m)) # Ks[i] = K_i
        Ms = np.zeros((self.process.num_steps, self.process.n, self.process.n)) # Ms[i] = M_i
        Sigmas_post[-1] = np.zeros((self.process.n, self.process.n)) # Strictly speaking, not necessary since it's already initialized to zero
        for i in range(self.process.num_steps):
            Sigmas[i] = self.process.A @ Sigmas_post[i-1] @ self.process.A.T + self.process.Sigma_process
            S_i = self.process.O @ Sigmas[i] @ self.process.O.T + self.process.Sigma_obs # Innovation covariance
            Ks[i] = Sigmas[i] @ self.process.O.T @ np.linalg.inv(S_i)
            Sigmas_post[i] = Sigmas[i] - Ks[i] @ self.process.O @ Sigmas[i]
            Ms[i] = self.process.A - Ks[i] @ self.process.O @ self.process.A
        
        Sigma_infty_dist = np.zeros(self.process.num_steps)
        K_infty_dist = np.zeros(self.process.num_steps)
        M_infty_dist = np.zeros(self.process.num_steps)
        for i in range(self.process.num_steps):
            Sigma_infty_dist[i] = np.linalg.norm(Sigmas[i] - self.Sigma_infty, ord='fro')
            K_infty_dist[i] = np.linalg.norm(Ks[i] - self.K_infty, ord='fro')
            M_infty_dist[i] = np.linalg.norm(Ms[i] - self.M_infty, ord='fro')
        
        return Sigma_infty_dist, K_infty_dist, M_infty_dist, Sigmas, Ks, Ms

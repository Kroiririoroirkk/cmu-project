import math
import matplotlib.pyplot as plt
import numpy as np

import utils


class HMProcess:
    """
    A class for representing a linear hidden Markov process with Gaussian noise.
    
    Attributes
    ----------
    rng : np.random.Generator
        The RNG used for generating random Gaussian noise
    n : int
        The dimensionality of the latent state
    m : int
        The dimensionality of the observations
    A : np.ndarray, shape (n, n)
        The transition matrix for the latent state
    x0 : np.ndarray, shape (n,)
        The initial latent state
    Sigma_process : np.ndarray, shape (n, n)
        The covariance of the process noise
    O : np.ndarray, shape (m, n)
        The observation matrix
    Sigma_obs : np.ndarray, shape (m, m)
        The covariance of the observation noise
    num_steps : int
        The number of steps to simulate for
    start_from : int
        The index that represents the first time point after
        the conclusion of the buy-in period

    Methods
    -------
    get_start_from()
        Calculate an appropriate length for the buy-in period.
    simulate()
        Simulate a hidden Markov process governed by linear dynamics.
    calc_loss(xhats, xs)
        Calculate the loss between the two input arrays.
    """
    
    def __init__(self, rng, A, x0, Sigma_process, O, Sigma_obs, num_steps, start_from=None):
        """
        Parameters
        ----------
        rng : np.random.Generator
            The RNG used for generating random Gaussian noise
        A : np.ndarray, shape (n, n)
            The transition matrix for the latent state
        x0 : np.ndarray, shape (n,)
            The initial latent state
        Sigma_process : np.ndarray, shape (n, n)
            The covariance of the process noise
        O : np.ndarray, shape (m, n)
            The observation matrix
        Sigma_obs : np.ndarray, shape (m, m)
            The covariance of the observation noise
        num_steps : int
            The number of steps to simulate for
        start_from : int
            The index that represents the first time point after
            the conclusion of the buy-in period (default automatically
            calculated)

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape, or
            if the length of the buy-in period is automatically calculated
            and the observation noise is over twice the process noise in
            standard deviation.
        """
        self.rng = rng
        self.n = A.shape[0]
        self.m = O.shape[0]
        self.A = A
        self.x0 = x0
        self.Sigma_process = Sigma_process
        self.O = O
        self.Sigma_obs = Sigma_obs
        self.num_steps = num_steps

        try:
            assert A.shape == (self.n, self.n)
            assert x0.shape == (self.n,)
            assert Sigma_process.shape == (self.n, self.n)
            assert O.shape == (self.m, self.n)
            assert Sigma_obs.shape == (self.m, self.m)
        except AssertionError:
            raise ValueError('Argument not of the correct shape')

        if start_from is None:
            self.start_from = self.get_start_from()
        else:
            self.start_from = start_from

    def get_start_from(self):
        """Calculate an appropriate length for the buy-in period.

        Returns
        -------
        start_from : int
            The index that represents the first time point after
            the conclusion of the buy-in period

        Raises
        ------
        ValueError
            If the observation noise is over twice the process noise
            in standard deviation.
        """
        try:
            sigma_process = utils.highest_eig(self.Sigma_process)
            sigma_obs = utils.highest_eig(self.Sigma_obs)
            assert sigma_obs <= 4 * sigma_process
        except AssertionError:
            raise ValueError('Observation noise is over twice the process noise')
        return math.ceil(utils.highest_eig(self.A)*11+2) # Rule of thumb from Exploration 3
    
    def simulate(self):
        """Simulate a hidden Markov process governed by linear dynamics.
    
        Returns
        -------
        ts : np.ndarray, shape (num_steps)
            The time values of the process
        xs : np.ndarray, shape (num_steps, n)
            The simulated latent states
        ys : np.ndarray, shape (num_steps, m)
            The simulated observations
        """
        xs = np.zeros((self.num_steps, self.n))
        ys = np.zeros((self.num_steps, self.m))
        xs[-1] = self.x0
        for i in range(self.num_steps):
            xs[i] = self.A@xs[i-1] + self.rng.multivariate_normal(np.zeros(self.n), self.Sigma_process)
            ys[i] = self.O@xs[i] + self.rng.multivariate_normal(np.zeros(self.m), self.Sigma_obs)
        return np.arange(self.num_steps), xs, ys

    def calc_loss(self, xhats, xs):
        """Calculate the loss between the two input arrays.
    
        More precisely, the loss is defined to be half of the
        sum-of-squares distance between the two arrays, starting
        from the index determined by the `start_from` value of
        the process.
    
        Arguments
        ---------
        xhats : np.ndarray, any shape
            The first input array
        xs : np.ndarray, same shape as xhats
            The second input array
    
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
        return np.sum((xhats[self.start_from:]-xs[self.start_from:])**2)/2


def plot_hm_process(title, ts, xs, ys, xhats=None):
    """Plot a hidden Markov process, optionally with an inferred latent state.

    Parameters
    ----------
    title : str
        The title of the plot
    ts : ndarray, shape (num_steps,)
        The time values of the process
    xs : ndarray, shape (num_steps, n)
        The latent states
    ys : ndarray, shape (num_steps, m)
        The observations
    xhats : Optional[ndarray, shape (num_steps, n)]
        The inferred latent states (default None)

    Raises
    ------
    ValueError
        If m != n (this function assumes the observation matrix is the identity),
        or if a NumPy array argument is not of the correct shape.
    """
    n = xs.shape[1]
    m = ys.shape[1]
    try:
        assert m == n
    except AssertionError:
        raise ValueError('Observation matrix is not the identity')
    try:
        num_steps = ts.shape[0]
        assert ts.shape == (num_steps,)
        assert xs.shape == (num_steps, n)
        assert ys.shape == (num_steps, m)
        if xhats is not None:
            assert xhats.shape == (num_steps, n)
    except AssertionError:
        raise ValueError('Argument not of the correct shape')

    fig, axs = plt.subplots(1, n, sharey=True, squeeze=False)
    axs = np.reshape(axs, (n,))
    fig.set_size_inches(12, 4.5)
    for i in range(n):
        axs[i].plot(ts, xs[:,i], '-', color='navy', label='Latent')
        axs[i].plot(ts, ys[:,i], '.', color='salmon', label='Observed')
        if xhats is not None:
            axs[i].plot(ts, xhats[:,i], '-', color='dodgerblue', label='Inferred')
        axs[i].set_title(f'Dimension {i+1}')
        axs[i].set_xlabel('Time')
    axs[0].set_ylabel('Value')
    fig.legend(*axs[0].get_legend_handles_labels(), loc='right', borderaxespad=0.15)
    fig.subplots_adjust(wspace=0.1)
    fig.suptitle(title, y=0.995)
    plt.show()

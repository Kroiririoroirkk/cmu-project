import matplotlib.pyplot as plt
import numpy as np


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

    Methods
    -------
    simulate()
        Simulate a hidden Markov process governed by linear dynamics.
    """
    
    def __init__(self, rng, A, x0, Sigma_process, O, Sigma_obs, num_steps):
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

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
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
        xs[0] = self.x0
        for i in range(1, self.num_steps):
            xs[i] = self.A@xs[i-1] + self.rng.multivariate_normal(np.zeros(self.n), self.Sigma_process)
        for i in range(self.num_steps):
            ys[i] = self.O@xs[i] + self.rng.multivariate_normal(np.zeros(self.m), self.Sigma_obs)
        return np.arange(self.num_steps), xs, ys
    
    
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

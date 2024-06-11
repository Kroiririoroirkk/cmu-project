import matplotlib.pyplot as plt
import numpy as np


def simulate_hm_process(rng, n, m, A, x0, Sigma_process, O, Sigma_obs, num_steps):
    """Simulate a hidden Markov process governed by linear dynamics.

    Parameters
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

    Returns
    -------
    ts : np.ndarray, shape (num_steps)
        The time values of the process
    xs : np.ndarray, shape (num_steps, n)
        The simulated latent states
    ys : np.ndarray, shape (num_steps, m)
        The simulated observations

    Raises
    ------
    ValueError
        If a NumPy array argument is not of the correct shape.
    """
    try:
        assert A.shape == (n, n)
        assert x0.shape == (n,)
        assert Sigma_process.shape == (n, n)
        assert O.shape == (m, n)
        assert Sigma_obs.shape == (m, m)
    except AssertionError:
        raise ValueError('Argument not of the correct shape')
    xs = np.zeros((num_steps, n))
    ys = np.zeros((num_steps, m))
    curr = x0
    for i in range(num_steps):
        xs[i] = curr
        ys[i] = O@curr + rng.multivariate_normal(np.zeros(m), Sigma_obs)
        curr = A@curr + rng.multivariate_normal(np.zeros(n), Sigma_process)
    return np.arange(num_steps), xs, ys


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

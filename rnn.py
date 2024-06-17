import numpy as np
import tqdm

from hm_process import HMProcess
from utils import calc_loss


class NeuralNet:
    """
    A class for representing a linear recurrent neural network

    More specifically, the task of the neural network is to take
    in noisy observations and filter them to infer a latent signal.

    Attributes
    ----------
    num_neurons : int
        The number of neurons
    latent_dim : int
        The dimensionality of the latent state
    obs_dim : int
        The dimensionality of the observations
    K : np.ndarray, shape (num_neurons, obs_dim)
        The input connections
    M : np.ndarray, shape (num_neurons, num_neurons)
        The connectivity matrix
    W : np.ndarray, shape (latent_dim, num_neurons)
        The readout matrix
    r0 : np.ndarray, shape (num_neurons,)
        The initial state of the neural network
    mask : np.ndarray, shape (num_neurons, num_neurons)
        A matrix of 0s and 1s that specifies the support of M

    Methods
    -------
    forward(ys)
        Infer a latent signal from the observations given.
    batch_test(self, batch_size, proc)
        Test the network on simulated data.
    backward(ys, xs, start_from)
        Calculate the loss and gradient from one labeled example.
    train_batch(eta, num_trials, proc, start_from, print_loss,
            progress_bar)
        Train the network on simulated data.
    train(etas, num_trials_per, proc, start_from, print_loss,
            progress_bar)
        Apply `train_batch` several times.
    train_until_converge(eta, epsilon, num_trials_per, proc, start_from,
            print_loss, progress_bar)
        Apply `train_batch` until the losses differ by less than `epsilon`.
    """

    def __init__(self, M0, K0, W0, r0, mask=None):
        """
        Parameters
        ----------
        M0 : np.ndarray, shape (num_neurons, num_neurons)
            The connectivity matrix
        K0 : np.ndarray, shape (num_neurons, obs_dim)
            The input connections
        W0 : np.ndarray, shape (latent_dim, num_neurons)
            The readout matrix
        r0 : np.ndarray, shape (num_neurons,)
            The initial state of the neural network
        mask : np.ndarray, shape (num_neurons, num_neurons)
            A matrix of 0s and 1s that specifies the support of M
            (default all 1s, indicated by a value of None)
        """
        self.M = M0
        self.K = K0
        self.W = W0
        self.r0 = r0
        self.num_neurons = self.r0.shape[0]
        self.latent_dim = self.W.shape[0]
        self.obs_dim = self.K.shape[1]
        if mask is None:
            self.mask = np.full((self.num_neurons, self.num_neurons), 1)
        else:
            self.mask = mask
        self.M *= self.mask

    def forward(self, ys):
        """Infer a latent signal from the observations given.

        Parameters
        ----------
        ys : np.ndarray, shape (num_steps, obs_dim)
            The array of observations

        Returns
        -------
        rs : np.ndarray, shape (num_steps, num_neurons)
            The sequence of neural states achieved
        xhats : np.ndarray, shape (num_steps, latent_dim)
            The inferred signal

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        """
        num_steps = ys.shape[0]
        try:
            assert ys.shape == (num_steps, self.obs_dim)
        except AssertionError:
            raise ValueError('Argument not of the correct shape')
        rs = np.zeros((num_steps, self.num_neurons))
        xhats = np.zeros((num_steps, self.latent_dim))

        rs[-1] = self.r0
        for i in range(num_steps):
            rs[i] = self.M @ rs[i-1] + self.K @ ys[i]
            xhats[i] = self.W @ rs[i]

        return rs, xhats

    def batch_test(self, batch_size, proc, start_from=0):
        """Test the network on simulated data.

        Parameters
        ----------
        batch_size : int
            The number of trials to simulate
        proc : HMProcess
            The hidden Markov process to simulate
        start_from : int
            The time index to start calculating loss from (default 0)

        Returns
        -------
        losses : np.ndarray, shape (batch_size,)
            The loss of each trial

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        """
        losses = np.zeros(batch_size)
        for i in range(batch_size):
            _, xs, ys = proc.simulate()
            _, xhats = self.forward(ys)
            losses[i] = calc_loss(xhats, xs, start_from)
        return losses

    def backward(self, ys, xs, start_from=0):
        """Calculate the loss and gradient from one labeled example.

        Parameters
        ----------
        ys : np.ndarray, shape (num_steps, obs_dim)
            The array of observations
        xs : np.ndarray, shape (num_steps, latent_dim)
            The true latent signal
        start_from : int
            The time index to start calculating loss from (default 0)

        Returns
        -------
        L : float
            The loss on the training example
        dL_dM : np.ndarray, shape (num_neurons, num_neurons)
            The derivative of the loss with respect to the
            connectivity matrix `M`, where `dL_dM[i,j]` represents
            the derivative of `L` with respect to `M[i,j]`
        dL_dK : np.ndarray, shape (num_neurons, obs_dim)
            The derivative of the loss with respect to the
            input matrix `K`, where `dL_dK[i,j]` represents the
            derivative of `L` with respect to `K[i,j]`

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        """
        num_steps = ys.shape[0]
        try:
            assert xs.shape == (num_steps, self.latent_dim)
        except AssertionError:
            raise ValueError('Argument not of the correct shape')
        rs, xhats = self.forward(ys)

        # dr_dr[m,n,k,l] represents dr^m_k / dr^n_l for m > n and otherwise 0
        dr_dr = np.zeros((num_steps, num_steps, self.num_neurons, self.num_neurons))
        for m in range(num_steps-1, -1, -1):
            for k in range(self.num_neurons):
                for l in range(self.num_neurons):
                    dr_dr[m,m-1,k,l] = self.M[k,l]
            for n in range(m-2, -1, -1):
                for k in range(self.num_neurons):
                    for l in range(self.num_neurons):
                        dr_dr[m,n,k,l] = sum(dr_dr[m,n+1,k,i]*self.M[i,l] for i in range(self.num_neurons))

        # dL_dr[n,k] represents dL / dr^n_k where L = 1/2 sum_{im} (xhat^m_i - x^m_i)^2
        dL_dr = np.zeros((num_steps, self.num_neurons))
        for n in range(start_from, num_steps):
            for k in range(self.num_neurons):
                dL_dr[n,k] = sum((self.W[i,k]*rs[n,k]-xs[n,i])*self.W[i,k] for i in range(self.latent_dim)) + sum(sum(sum(
                    (self.W[i,j]*rs[m,j]-xs[m,i])*self.W[i,j]*dr_dr[m,n,j,k]
                    for i in range(self.latent_dim))
                    for j in range(self.num_neurons))
                    for m in range(n+1,num_steps))
        for n in range(start_from):
            for k in range(self.num_neurons):
                dL_dr[n,k] = sum(sum(sum(
                    (self.W[i,j]*rs[m,j]-xs[m,i])*self.W[i,j]*dr_dr[m,n,j,k]
                    for i in range(self.latent_dim))
                    for j in range(self.num_neurons))
                    for m in range(start_from,num_steps))

        kron_delta = np.eye(self.num_neurons)

        # dr_dM[n,k,i,l] represents dr^n_k / dM_il
        rs_shift = np.roll(rs, 1, 0)
        rs_shift[0,:] = 0
        dr_dM = np.einsum('jk,il', kron_delta, rs_shift, optimize='greedy')
        #dr_dM = np.zeros((num_steps, self.num_neurons, self.num_neurons, self.num_neurons))
        #for n in range(1,num_steps):
        #    for k in range(self.num_neurons):
        #        for i in range(self.num_neurons):
        #            for l in range(self.num_neurons):
        #                dr_dM[n,k,i,l] = rs[n-1,l] if i==k else 0

        # dr_dK[n,k,i,l] represents dr^n_k / dK_il
        dr_dK = np.einsum('jk,il', kron_delta, ys, optimize='greedy')
        #dr_dK = np.zeros((num_steps, self.num_neurons, self.num_neurons, self.obs_dim))
        #for n in range(num_steps):
        #    for k in range(self.num_neurons):
        #        for i in range(self.num_neurons):
        #            for l in range(self.obs_dim):
        #                dr_dK[n,k,i,l] = ys[n,l] if i==k else 0

        # dL_dM[i,j] represents dL / dM_ij
        dL_dM = np.einsum('ij,ijkl', dL_dr, dr_dM, optimize='greedy')
        #dL_dM = np.zeros((self.num_neurons, self.num_neurons))
        #for i in range(self.num_neurons):
        #    for j in range(self.num_neurons):
        #        dL_dM[i,j] = sum(sum(dL_dr[n,k]*dr_dM[n,k,i,j] for n in range(num_steps)) for k in range(self.num_neurons))

        # dL_dK[i,j] represents dL / dK_ij
        dL_dK = np.einsum('ij,ijkl', dL_dr, dr_dK, optimize='greedy')
        #dL_dK = np.zeros((self.num_neurons, self.obs_dim))
        #for i in range(self.num_neurons):
        #    for j in range(self.obs_dim):
        #        dL_dK[i,j] = sum(sum(dL_dr[n,k]*dr_dK[n,k,i,j] for n in range(num_steps)) for k in range(self.num_neurons))

        L = calc_loss(xhats, xs)
        return L, dL_dM, dL_dK

    def train_batch(self, eta, num_trials, proc, start_from=0, print_loss=True, progress_bar=True):
        """Train the network on simulated data.

        Parameters
        ----------
        eta : float
            The learning rate
        num_trials : int
            The number of trials to simulate
        proc : HMProcess
            The hidden Markov process to simulate
        start_from : int
            The time index to start calculating loss from (default 0)
        print_loss : bool
            Whether to print the mean loss (default True)
        progress_bar : bool
            Whether to print a progress bar (default True)

        Returns
        -------
        losses : np.ndarray, shape (num_trials,)
            The loss of each trial
        dL_dMs : np.ndarray, shape (num_trials, num_neurons, num_neurons)
            The value of `dL_dM` for each trial (see the documentation
            for `backward`)
        dL_dKs : np.ndarray, shape (num_trials, num_neurons, obs_dim)
            The value of `dL_dK` for each trial (see the documentation
            for `backward`)

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        """
        losses = np.zeros(num_trials)
        dL_dMs = np.zeros((num_trials, self.num_neurons, self.num_neurons))
        dL_dKs = np.zeros((num_trials, self.num_neurons, self.obs_dim))

        r = range(num_trials)
        if progress_bar:
            r = tqdm.tqdm(r)
        for i in r:
            _, xs, ys = proc.simulate()
            L, dL_dM, dL_dK = self.backward(ys, xs, start_from)
            losses[i] = L
            dL_dMs[i] = dL_dM
            dL_dKs[i] = dL_dK

        dL_dM_mean = np.mean(dL_dMs, axis=0)
        dL_dK_mean = np.mean(dL_dKs, axis=0)
        self.M -= eta * dL_dM_mean * self.mask
        self.K -= eta * dL_dK_mean

        if print_loss:
            print('Mean loss', np.mean(losses))

        return losses, dL_dMs, dL_dKs

    def train(self, etas, num_trials_per, proc, start_from=0, print_loss=True, progress_bar=True):
        """Apply `train_batch` several times.

        Parameters
        ----------
        etas : np.ndarray, shape (num_batches,)
            The sequence of learning rates, one for each batch
        num_trials_per : int
            The number of trials to simulate for each batch
        proc : HMProcess
            The hidden Markov process to simulate
        start_from : int
            The time index to start calculating loss from (default 0)
        print_loss : bool
            Whether to print the mean loss (default True)
        progress_bar : bool
            Whether to print a progress bar (default True)

        Returns
        -------
        losses : np.ndarray, shape (num_batches,)
            The mean loss for each batch

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        """
        num_batches = etas.shape[0]
        losses = np.zeros(num_batches)
        for i in range(num_batches):
            Ls, _, _ = self.train_batch(etas[i], num_trials_per, proc, start_from, print_loss, progress_bar)
            losses[i] = np.mean(Ls)
        return losses

    def train_until_converge(self, eta, epsilon, num_trials_per, proc, start_from=0, print_loss=True, progress_bar=True):
        """Apply `train_batch` until the losses differ by less than `epsilon`.

        Parameters
        ----------
        eta : float
            The learning rate
        epsilon : float
            The threshold for determining if two losses are the same
        num_trials_per : int
            The number of trials to simulate for each batch
        proc : HMProcess
            The hidden Markov process to simulate
        start_from : int
            The time index to start calculating loss from (default 0)
        print_loss : bool
            Whether to print the mean loss (default True)
        progress_bar : bool
            Whether to print a progress bar (default True)

        Returns
        -------
        num_batches : int
            The number of batches
        losses : np.ndarray, shape (num_batches,)
            The mean loss for each batch
        Ms : np.ndarray, shape (num_batches+1, num_neurons, num_neurons)
            The values of M achieved during training
        Ks : np.ndarray, shape (num_batches+1, num_neurons, obs_dim)
            The values of K achieved during training

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        """
        Ms = [np.copy(self.M)]
        Ks = [np.copy(self.K)]
        mean_losses = []
        
        def iter_loop():
            Ls, _, _ = self.train_batch(eta, num_trials_per, proc, start_from, print_loss, progress_bar)
            Ms.append(np.copy(self.M))
            Ks.append(np.copy(self.K))
            mean_losses.append(np.mean(Ls))
        
        iter_loop() # This guarantees we have at least two entries before we check whether to break the loop
        while True:
            iter_loop()
            if abs(mean_losses[-1] - mean_losses[-2]) < epsilon:
                break
        
        return len(mean_losses), np.array(mean_losses), np.array(Ms), np.array(Ks)

from enum import Enum
import numpy as np
import tqdm

from hm_process import HMProcess


Param = Enum('Param', ['M', 'K', 'W'])
"""
A class for representing the possible learnable parameters in a neural network
"""


class Nonlinearity(Enum):
    """
    A class for representing the possible nonlinearities present in a neural network

    Attributes
    ----------
    fn : Callable[float, float]
        Function that applies the nonlinearity (should also work on NumPy arrays)
    deriv : Callable[float, float]
        Function that applies the derivative of the nonlinearity (should also
        work on NumPy arrays)
    """
    ID = (lambda x: x), (lambda x: 0*x+1)
    TANH = (lambda x: 50*(1+np.tanh(x))), (lambda x: 50/(np.cosh(x)**2))
    RELU = (lambda x: (x>0)*x), (lambda x: (x>0)*1)

    def __init__(self, fn, deriv):
        """
        Parameters
        ----------
        fn : Callable[float, float]
            Function that applies the nonlinearity (should also work on NumPy arrays)
        deriv : Callable[float, float]
            Function that applies the derivative of the nonlinearity (should also
            work on NumPy arrays)
        """
        self.fn = fn
        self.deriv = deriv


class NeuralNet:
    """
    A class for representing a recurrent neural network

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
    M : np.ndarray, shape (num_neurons, num_neurons)
        The connectivity matrix
    K : np.ndarray, shape (num_neurons, obs_dim)
        The input connections
    W : np.ndarray, shape (latent_dim, num_neurons)
        The readout matrix
    r0 : np.ndarray, shape (num_neurons,)
        The initial state of the neural network
    mask : np.ndarray, shape (num_neurons, num_neurons)
        A matrix of 0s and 1s that specifies the support of M
    train_vars : set[Param]
        Which parameters should be modified during training
    nonlin : Nonlinearity
        The nonlinearity used by the neural network

    Methods
    -------
    forward(ys)
        Infer a latent signal from the observations given.
    batch_test(self, batch_size, proc)
        Test the network on simulated data.
    backward(ys, xs, proc)
        Calculate the loss and gradient from one labeled example.
    train_batch(eta, num_trials, proc, print_loss, progress_bar)
        Train the network on simulated data.
    train(etas, num_trials_per, proc, print_loss, progress_bar)
        Apply `train_batch` several times.
    train_until_converge(eta, epsilon, num_trials_per, proc,
            print_loss, progress_bar)
        Apply `train_batch` until the losses differ by less than `epsilon`.
    """

    def __init__(self, M0, K0, W0, r0, mask=None, train_vars=None, nonlin=Nonlinearity.ID):
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
        train_vars : set[Param]
            Which parameters should be modified during training
        nonlin : Nonlinearity
            The nonlinearity used by the neural network
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
        if train_vars is None:
            self.train_vars = {Param.M, Param.K, Param.W}
        else:
            self.train_vars = train_vars
        self.nonlin = nonlin

    def forward(self, ys):
        """Infer a latent signal from the observations given.

        Parameters
        ----------
        ys : np.ndarray, shape (num_steps, obs_dim)
            The array of observations

        Returns
        -------
        ss : np.ndarray, shape (num_steps, num_neurons)
            The sequence of neural states achieved, pre-nonlinearity
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
        ss = np.zeros((num_steps, self.num_neurons))
        rs = np.zeros((num_steps, self.num_neurons))
        xhats = np.zeros((num_steps, self.latent_dim))

        rs[-1] = self.r0
        for i in range(num_steps):
            ss[i] = self.M @ rs[i-1] + self.K @ ys[i]
            rs[i] = self.nonlin.fn(ss[i])
            xhats[i] = self.W @ rs[i]

        return ss, rs, xhats

    def batch_test(self, batch_size, proc):
        """Test the network on simulated data.

        Parameters
        ----------
        batch_size : int
            The number of trials to simulate
        proc : HMProcess
            The hidden Markov process to simulate

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
            _, _, xhats = self.forward(ys)
            losses[i] = proc.calc_loss(xhats, xs)
        return losses

    def backward(self, ys, xs, proc):
        """Calculate the loss and gradient from one labeled example.

        Parameters
        ----------
        ys : np.ndarray, shape (num_steps, obs_dim)
            The array of observations
        xs : np.ndarray, shape (num_steps, latent_dim)
            The true latent signal
        proc : HMProcess
            The hidden Markov process the example came from

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
        dL_dW : np.ndarray, shape (latent_dim, num_neurons)
           The derivative of the loss with respect to the
            readout matrix `W`, where `dL_dW[i,j]` represents the
            derivative of `L` with respect to `W[i,j]`

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        OverflowError
            If the loss blows up.
        """
        num_steps = ys.shape[0]
        try:
            assert xs.shape == (num_steps, self.latent_dim)
        except AssertionError:
            raise ValueError('Argument not of the correct shape')
        ss, rs, xhats = self.forward(ys)
        rs_shift = np.roll(rs, 1, axis=0)
        rs_shift[0] = self.r0

        with np.errstate(over='raise'):
            dL_dW = np.tensordot(xhats[proc.start_from:]-xs[proc.start_from:],
                                 rs[proc.start_from:],
                                 axes=([0],[0]))
            dL_dr = np.zeros((proc.num_steps, self.num_neurons))
            dL_dr[-1] = (xhats[-1]-xs[-1]) @ self.W
            for i in range(-2,-1*proc.num_steps-1,-1):
                dL_dr[i] = ((xhats[i]-xs[i]) @ self.W
                    + (dL_dr[i+1]*self.nonlin.deriv(ss[i+1])) @ self.M)
            dL_dr_phi = dL_dr * self.nonlin.deriv(ss)
            dL_dM = np.tensordot(dL_dr_phi, rs_shift, axes=([0],[0]))
            dL_dK = np.tensordot(dL_dr_phi, ys, axes=([0],[0]))
    
            L = proc.calc_loss(xhats, xs)
            return L, dL_dM, dL_dK, dL_dW

    def train_batch(self, eta, num_trials, proc, print_loss=True, progress_bar=True):
        """Train the network on simulated data.

        Parameters
        ----------
        eta : float
            The learning rate
        num_trials : int
            The number of trials to simulate
        proc : HMProcess
            The hidden Markov process to simulate
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
        dL_dWs : np.ndarray, shape (num_trials, latent_dim, num_neurons)
            The value of `dL_dW` for each trial (see the documentation
            for `backward`)

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        OverflowError
            If the loss blows up.
        """
        losses = np.zeros(num_trials)
        dL_dMs = np.zeros((num_trials, self.num_neurons, self.num_neurons))
        dL_dKs = np.zeros((num_trials, self.num_neurons, self.obs_dim))
        dL_dWs = np.zeros((num_trials, self.latent_dim, self.num_neurons))

        r = range(num_trials)
        if progress_bar:
            r = tqdm.tqdm(r)
        for i in r:
            _, xs, ys = proc.simulate()
            L, dL_dM, dL_dK, dL_dW = self.backward(ys, xs, proc)
            losses[i] = L
            if L > 1e10:
                raise OverflowError
            dL_dMs[i] = dL_dM
            dL_dKs[i] = dL_dK
            dL_dWs[i] = dL_dW

        if Param.M in self.train_vars:
            dM = eta * np.mean(dL_dMs, axis=0) * self.mask
            dM[dM>0.1] = 0.1
            dM[dM<-0.1] = -0.1
            self.M -= dM
        if Param.K in self.train_vars:
            dK = eta * np.mean(dL_dKs, axis=0)
            dK[dK>0.1] = 0.1
            dK[dK<-0.1] = -0.1
            self.K -= dK
        if Param.W in self.train_vars:
            dW = eta * np.mean(dL_dWs, axis=0)
            dW[dW>0.1] = 0.1
            dW[dW<-0.1] = -0.1
            self.W -= dW

        if print_loss:
            print('Mean loss', np.mean(losses))

        return losses, dL_dMs, dL_dKs, dL_dWs

    def train(self, etas, num_trials_per, proc, print_loss=True, progress_bar=True):
        """Apply `train_batch` several times.

        Parameters
        ----------
        etas : np.ndarray, shape (num_batches,)
            The sequence of learning rates, one for each batch
        num_trials_per : int
            The number of trials to simulate for each batch
        proc : HMProcess
            The hidden Markov process to simulate
        print_loss : bool
            Whether to print the mean loss (default True)
        progress_bar : bool
            Whether to print a progress bar (default True)

        Returns
        -------
        losses : np.ndarray, shape (num_batches,)
            The mean loss for each batch
        Ms : np.ndarray, shape (num_batches+1, num_neurons, num_neurons)
            The values of M achieved during training
        Ks : np.ndarray, shape (num_batches+1, num_neurons, obs_dim)
            The values of K achieved during training
        Ws : np.ndarray, shape (num_batches+1, latent_dim, num_neurons)
            The values of W achieved during training

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        OverflowError
            If the loss blows up.
        """
        num_batches = etas.shape[0]
        Ms = np.zeros((num_batches+1, self.num_neurons, self.num_neurons))
        Ms[0] = self.M
        Ks = np.zeros((num_batches+1, self.num_neurons, self.obs_dim))
        Ks[0] = self.K
        Ws = np.zeros((num_batches+1, self.latent_dim, self.num_neurons))
        Ws[0] = self.W
        losses = np.zeros(num_batches)
        r = range(num_batches)
        if progress_bar:
            r = tqdm.tqdm(r)
        for i in r:
            Ls, _, _, _ = self.train_batch(etas[i], num_trials_per, proc, print_loss, progress_bar=False)
            losses[i] = np.mean(Ls)
            Ms[i+1] = self.M
            Ks[i+1] = self.K
            Ws[i+1] = self.W
        return losses, Ms, Ks, Ws

    def train_until_converge(self, eta, epsilon, num_trials_per, proc, print_loss=True, progress_bar=True):
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
        Ws : np.ndarray, shape (num_batches+1, latent_dim, num_neurons)
            The values of W achieved during training

        Raises
        ------
        ValueError
            If a NumPy array argument is not of the correct shape.
        OverflowError
            If the loss blows up.
        """
        Ms = [np.copy(self.M)]
        Ks = [np.copy(self.K)]
        Ws = [np.copy(self.W)]
        mean_losses = []
        
        def iter_loop():
            Ls, _, _, _ = self.train_batch(eta, num_trials_per, proc, print_loss, progress_bar)
            Ms.append(np.copy(self.M))
            Ks.append(np.copy(self.K))
            Ws.append(np.copy(self.W))
            mean_losses.append(np.mean(Ls))
        
        iter_loop() # This guarantees we have at least two entries before we check whether to break the loop
        while True:
            iter_loop()
            if abs(mean_losses[-1] - mean_losses[-2]) < epsilon:
                break
        
        return len(mean_losses), np.array(mean_losses), np.array(Ms), np.array(Ks), np.array(Ws)

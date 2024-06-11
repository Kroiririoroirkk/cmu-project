# CMU Summer 2024 uPNC Project

Eric Tao, 2024-06-10

## Introduction

This is the repository I (Eric Tao) have made for the code I have written as part of the Summer 2024 uPNC (Undergraduate Program in Neural Computation) at CMU (Carnegie Mellon University). The aim of this project is to investigate the functional basis of why neurons are connected in the patterns that they are. For example, [previous work](https://bmcbiol.biomedcentral.com/articles/10.1186/1741-7007-2-25) by Reigl et al., 2004 has found that certain types of network motifs such as bi-directionally connected pairs of neurons and transitive triangles of neurons are overrepresented inside a _C. elegans_ brain. How does the presence of these motifs in a neural network change its ability to perform various tasks? What tasks do networks with these motifs excel at? What tasks do networks with these motifs fail at? The answers to these questions would yield insight into the evolutionary goals of the brain and bridge the gap between network connectivity and functionality, analogous to Marr's implementational and computational levels.

The main notebook file for my exploration is `LDS_exploration.ipynb`, which depends on the `.py` files in the parent directory. For posterity, I have also included some old unused code in the `old` folder which uses PyTorch. I have since decided to code the system from scratch for a better understanding of the internals and more flexibility.

## Setup

### Task

Suppose we have hidden variable states $x_0, x_1, x_2, \dots \in \mathbb R^n$ which evolve over time according to a linear dynamics rule $x_t = Ax_{t-1} + w_{t-1}$, where $A \in \mathbb R^{n\times n}$ is the matrix determining the dynamics and $w_t \sim \mathcal N(0^n, \Sigma_\text{process}^{n\times n})$ is IID process noise.

At each time $t$, we then make an observation $y_t \in \mathbb R^m$ which is determined by $x_t$ according to $y_t = Ox_t + v_t$ for some observation matrix $O \in \mathbb R^{m\times n}$ and some IID observation noise $v_t \sim \mathcal N(0^m, \Sigma_\text{obs}^{m\times m})$. For simplicity, we shall take $m=n$ and $O$ to be the identity matrix.

In other words, we have a hidden Markov process where both the state transitions and observations are linear, with Gaussian noise.

The task we would like to accomplish is to take in a series of observations $y_0, \dots, y_T$ as an input and use them to calculate an output $\hat x_0, \dots, \hat x_T$, minimizing the squared error $\mathcal L = \sum_{t=0}^T \|x_t - \hat x_t\|_2^2$.

### Theory

The optimal estimator for this task (in the sense of minimizing mean squared error) is known as the Kalman filter and leverages Bayesian inference to combine a prior guess $P(x_t \mid y_1,\dots, y_{t-1})$ with a likelihood $P(y_t \mid x_t)$ to obtain a posterior guess $P(x_t \mid y_1,\dots,y_t)$. The result is that the mean of the posterior distribution will be a linear combination of the guess resulting from the mean of the prior distribution and our new observation:

$$\hat x_t = A\hat x_{t-1} + K_t(y_t - OA\hat x_{t-1}),$$

where $K_t \in \mathbb R^{n\times m}$ is known as the Kalman gain matrix and represents how much we trust our new observation compared to how much we trust our previous guess.

The Kalman gain has its own evolution equation (which depends on the system parameters $A,\Sigma_\text{process},O,\Sigma_\text{obs}$, but not on $x_t$, $y_t$, or $\hat x_t$), but the steady-state value of the Kalman gain $K_\infty$ (which $K_t$ converges to under some reasonable assumptions) is given by the equation

$$K_\infty = \Sigma_\infty O^T (\Sigma_\text{obs} + O\Sigma_\infty O^T)^{-1},$$

where $\Sigma_\infty \in \mathbb R^{n\times n}$ is the steady-state covariance of the estimated latent state and can be calculated by solving the discrete Riccati equation

$$\Sigma_\infty = A(\Sigma_\infty - \Sigma_\infty O^T(O\Sigma_\infty O^T + \Sigma_\text{obs})^{-1}O\Sigma_\infty)A^T + \Sigma_\text{process}.$$

Since we are taking $O$ to be the identity, we can simplify this to obtain

$$K_\infty = \Sigma_\infty (\Sigma_\text{obs} + \Sigma_\infty)^{-1},$$

and

$$\Sigma_\infty = A(\Sigma_\infty - \Sigma_\infty(\Sigma_\infty + \Sigma_\text{obs})^{-1}\Sigma_\infty)A^T + \Sigma_\text{process}.$$

For more information, see [this tutorial](https://compneuro.neuromatch.io/tutorials/W3D2_HiddenDynamics/student/W3D2_Tutorial3.html) introducing the Kalman filter and [these lecture slides](https://www.cs.cmu.edu/~motionplanning/papers/sbp_papers/kalman/kleeman_understanding_kalman.pdf).

### Model

Drawing inspiration from a neural net, we would like to compute the optimal output to the task using a set of $\ell$ neurons with internal states of $r = (r_1, \dots, r_\ell)$, whose dynamics are governed by the equation $r_t = Mr_{t-1} + Ky_t$ (here, $K$ is analogous to but not the same as the Kalman gain above), where $M \in \mathbb R^{\ell\times\ell}$ and $K \in \mathbb R^{\ell\times m}$. That is, the internal states of the neurons will update according to both the state of the network at the previous time step as well as the new information. Then, our prediction will be a linear function of the neural states: $\hat x_t = Wr_t$, where $W \in \mathbb R^{n\times\ell}$.

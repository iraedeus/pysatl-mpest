# Usage

File `src/em.py` contains class `EM`.
Method `EM.em_algo` contains static realization of em algo for solving the parameter estimation of distributions mixture problem:

$$p(x | \Omega, F, \Theta) = \sum_{j=1}^k \omega_j f_j(x | \theta_j)$$

- $p(x | \Omega, F, \Theta)$ - distributions mixture
- $f_j(x | \theta_j)$ - $j$ distribution
- $\theta_j$ - parameters of $j$ distribution
- $\omega_j$ - prior probability of $j$ distribution

## Requirements
- python 3.11
- libraries, which listed in `requirements` file

# Usage

File `src/em.py` contains class `EM`.
Method `EM.em_algo` contains static realization of em algo for solving the parameter estimation of distributions mixture problem:

$$p(x | \Omega, F, \Theta) = \sum_{j=1}^k \omega_j f_j(x | \theta_j)$$

- $p(x | \Omega, F, \Theta)$ - distributions mixture
- $f_j(x | \theta_j)$ - $j$ distribution
- $\theta_j$ - parameters of $j$ distribution
- $\omega_j$ - prior probability of $j$ distribution

`EM.em_algo` takes params:
- `X` - sample of distributions mixture
- `distributions` - the starting guess of which distributions in distributions mixture
- `k` - guess of distributions count
- `deviation` - the error: max difference between each parameter of each distribution
- `max_step` - max count of EM algorithm iterations, `None` if not limited
- `prior_probability_threshold` - when prior_probability of distribution will become less then this parameter, this distribution will no longer be considered, `None` if not limited
- `prior_probability_threshold_step` - iteration after which `prior_probability_threshold` will be used
- `optimizer` - minimizer which will be used in M step of algo

## Requirements
- python 3.11
- libraries, which listed in `requirements` file

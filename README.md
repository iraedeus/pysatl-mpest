![PyLint](https://github.com/ToxaKaz/EM-algo/actions/workflows/pylint.yml/badge.svg)
![Check code style](https://github.com/ToxaKaz/EM-algo/actions/workflows/code_style.yml/badge.svg)
[![Code style](https://img.shields.io/badge/Code%20style-black-000000.svg)](https://github.com/psf/black)
![Unit Tests](https://github.com/ToxaKaz/EM-algo/actions/workflows/test.yml/badge.svg)

# Usage

This package contains realization of em algorithm for solving the parameter estimation of mixture distribution problem:

$$p(x | \Omega, F, \Theta) = \sum_{j=1}^k \omega_j f_j(x | \theta_j)$$

- $p(x | \Omega, F, \Theta)$ - mixture distribution
- $f_j(x | \theta_j)$ - $j$ distribution
- $\theta_j$ - parameters of $j$ distribution
- $\omega_j$ - prior probability of $j$ distribution

It can work with mixture distribution of any combination of models, which implements `AModel` class. EM algorithm result can be calculated by using `EM` class with `AOptimizer` implementation and guessed or random initial approximation.

## Requirements
- python 3.11
- libraries, which listed in `requirements` file

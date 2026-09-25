### Python Optimization Problem Toolbox (POPT)

POPT is one part of the PET application. Here we solve the optimization problem. 
Currently, the following methods are implemented: 

- EnOpt: The standard ensemble optimization method
- GenOpt: Generalized ensemble optimization (using non-Gaussian distributions)
- SmcOpt: Gradient-free optimization based on sequential Monte Carlo
- LineSearch: Gradient based method satisfying the strong Wolfe conditions
- TrustRegion: Trust-region method with an ensemble-built model

The gradient and Hessian methods are compatible with SciPy, and can be used as input to scipy.optimize.minimize. 
A POPT tutorial is found [here](https://python-ensemble-toolbox.github.io/PET/tutorials/popt/5Spot/tutorial_popt).

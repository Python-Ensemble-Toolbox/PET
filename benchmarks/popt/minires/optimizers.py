"""The methods: what is run on each case, and with which settings.

``kind`` selects how :func:`run.experiment` runs the entry:

- ``"ensqp"``: :class:`popt.EnSQP` on the objective that returns its constraints;
- ``"enopt"``: :class:`popt.EnOpt` on the objective plus the exterior penalty (``epf`` in the
  options), or on the objective alone (no ``epf``);
- ``"reference"``: scipy's SLSQP with finite-difference derivatives on the simulator, which is
  not an ensemble method and costs far more simulations; its result is what the others'
  net present value is measured against.

Every ensemble method works in the unit cube (``transform``), from the same start, with the same
perturbation size and the same ensemble seed.
"""

COMMON = {"transform": True, "maxiter": 40, "logit": False, "restart": False, "restartsave": False, "saveit": False}

ENOPT = {"alpha": 0.1, "alpha_maxiter": 5, "resample": 2, "normalize": True, "optimizer": "GD"}
EPF = {"r_factor": 4.0, "tol_factor": 0.9, "conv_crit": 1e-6, "max_epf_iter": 6}

METHODS = [
    {
        "label": "EnSQP (BFGS)",
        "kind": "ensqp",
        "options": {"hessian": "bfgs", "resample": 2, "max_step": 0.25},
    },
    {
        "label": "EnOpt + EPF (r = 10)",
        "kind": "enopt",
        "options": {**ENOPT, "maxiter": 15, "epf": {"r": 10.0, **EPF}},
    },
    {
        # A failed step raises the penalty factor instead of ending the run
        "label": "EnOpt + EPF (r = 10, advance on failure)",
        "kind": "enopt",
        "options": {**ENOPT, "maxiter": 15, "epf": {"r": 10.0, **EPF, "advance_on_failure": True}},
    },
    {
        "label": "EnOpt, constraints ignored",
        "kind": "enopt",
        "options": {**ENOPT},
    },
    {
        "label": "SLSQP, finite differences (reference)",
        "kind": "reference",
        "options": {"maxiter": 100, "ftol": 1e-9, "eps": 1e-4},
    },
]

REFERENCE = "SLSQP, finite differences (reference)"

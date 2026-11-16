"""The methods compared: each a label and its ``[dataassim]`` settings.

The settings are PET's own, as a config would give them, except that a distance localization
gives its ``radius`` in metres: ``run.py`` places a taper of that radius at each datum's well,
on the case's grid. Unless a row says it is tuned, the settings are defaults or the textbook
choice, the same for every case.
"""

COMMON = {"energy": 99.0, "obsname": "steps"}

METHODS = [
    {
        "label": "ES (one update)",
        "dataassim": {"scheme": "es", "analysis": "approx"},
    },
    {
        # The iterating schemes read their truncation from the iteration block, not from COMMON
        "label": "LM-EnRML (4 iterations)",
        "dataassim": {"scheme": "lmenrml", "analysis": "approx", "iteration": {"max_iter": 4, "energy": 99.0}},
    },
    {
        "label": "GN-EnRML (4 iterations)",
        "dataassim": {"scheme": "gnenrml", "analysis": "approx", "iteration": {"max_iter": 4, "energy": 99.0}},
    },
    {
        "label": "GN-EnRML (4 iterations), subspace2 analysis",
        "dataassim": {"scheme": "gnenrml", "analysis": "subspace2", "iteration": {"max_iter": 4, "energy": 99.0}},
    },
    {
        "label": "GN-EnRML (4 iterations), margis analysis",
        "dataassim": {"scheme": "gnenrml", "analysis": "margis", "iteration": {"max_iter": 4, "energy": 99.0}},
    },
    {
        "label": "ES-MDA (4 steps, α = 4)",
        "dataassim": {
            "scheme": "esmda",
            "analysis": "approx",
            "mda": {"tot_assim_steps": 4, "inflation_param": [4, 4, 4, 4]},
        },
    },
    {
        "label": "ES-MDA (4 steps, α = 4), enif analysis",
        "dataassim": {"scheme": "esmda", "analysis": "enif", "mda": {"tot_assim_steps": 4, "inflation_param": [4, 4, 4, 4]}},
    },
    {
        "label": "ES-MDA (4 steps, α = 4), adaptive localization (universal threshold)",
        "dataassim": {
            "scheme": "esmda",
            "analysis": "approx",
            "mda": {"tot_assim_steps": 4, "inflation_param": [4, 4, 4, 4]},
            "localization": {"name": "autoadaloc", "threshold": "universal", "type": "hard"},
        },
    },
    {
        "label": "ES-MDA (4 steps, α = 4), adaptive localization (3σ threshold)",
        "dataassim": {
            "scheme": "esmda",
            "analysis": "approx",
            "mda": {"tot_assim_steps": 4, "inflation_param": [4, 4, 4, 4]},
            "localization": {"name": "autoadaloc", "threshold": "adaptive", "cutoff": 3.0, "type": "hard"},
        },
    },
    {
        # The radius, in metres, is the prior's correlation range; the scalars have no position
        "label": "ES-MDA (4 steps, α = 4), distance localization (Gaspari-Cohn, 300 m)",
        "dataassim": {
            "scheme": "esmda",
            "analysis": "approx",
            "mda": {"tot_assim_steps": 4, "inflation_param": [4, 4, 4, 4]},
            "localization": {"name": "distance_loc", "taper_func": "gc", "radius": 300.0, "unlocalized": ["sor"]},
        },
    },
]

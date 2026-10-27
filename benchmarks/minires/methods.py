"""The methods compared: each a label, the question it belongs to, and its ``[dataassim]`` settings.

The settings are PET's own, as a config would give them. Unless a row says it is tuned,
they are defaults or the textbook choice, the same for every case.
"""

COMMON = {"energy": 99.0, "obsname": "steps"}

METHODS = [
    {
        "label": "ES-MDA, 4 x 4",
        "group": "Iterating",
        "dataassim": {
            "scheme": "esmda",
            "analysis": "approx",
            "mda": {"tot_assim_steps": 4, "inflation_param": [4, 4, 4, 4]},
        },
    },
]

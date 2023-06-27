"""Descriptive description."""

# External imports
from fnmatch import filter  # to check if wildcard name is in list
from importlib import import_module

def init_da(da_input,fwd_input,sim):
    "initialize the ensemble object based on the DA inputs"

    assert len(da_input['daalg']) == 2, f"Need to input assimilation type and update method, got {da_input['daalg']}"

    da_import = getattr(import_module('pipt.update_schemes.' + da_input['daalg'][0]), f'{da_input["daalg"][1]}_{da_input["analysis"]}')

    # Init. update scheme class, and get an object of that class
    return da_import(da_input, fwd_input,sim)

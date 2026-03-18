"""Common definitions for all import filters

Examples
--------
>>> from .ecl_common import Phase, Prop
"""


# Enumeration of the phases that can exist in a simulation.
#
# The names are abbreviations and should really be called oleic, aqueous and
# gaseous, if we wanted to be correct and type more.
#
# The value that is returned is the Eclipse moniker, for backwards
# compatibility with code that uses this directly.
Phase = type('Phase', (object,), {
    'oil': 'OIL',
    'wat': 'WAT',
    'gas': 'GAS',
})


# Properties that can be queried from an output file.
#
# The property forms the first part of a "selector", which is a tuple
# containing the necessary information to setup the name of the property
# to read.
Prop = type('Prop', (object,), {
    'pres': 'P',  # pressure
    'sat': 'S',   # saturation
    'mole': 'x',  # mole fraction
    'dens': 'D',  # density
    'temp': 'T',  # temperature
    'leak': 'L',  # leaky well
})

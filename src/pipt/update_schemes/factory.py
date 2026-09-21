"""Friendly constructors for the assimilation schemes.

The analysis flavour is a *parameter* of the algorithm, not a different
algorithm, so this module exposes one constructor per algorithm and takes the
flavour as an argument::

    from pipt import ESMDA
    scheme = ESMDA(cfg_da, cfg_en, sim, analysis="approx")

This mirrors how ``popt`` exposes ``EnOpt``/``LineSearch``/``TrustRegion`` as
one name per algorithm. The underlying concrete classes are unchanged and stay
importable, so ``isinstance`` checks and subclassing still work; these
constructors resolve through :mod:`pipt.update_schemes.registry` and return an
instance of exactly the same class as before.
"""

from pipt.update_schemes.registry import get_scheme

__all__ = ["EnKF", "ES", "ESMDA", "LMEnRML", "GNEnRML", "build_scheme"]


def build_scheme(scheme, da_input, en_input, sim, analysis=None):
    """Construct any registered scheme by name.

    Parameters
    ----------
    scheme : str
        Algorithm name, e.g. ``"esmda"``.
    da_input : dict
        Parsed data-assimilation config.
    en_input : dict
        Parsed ensemble config.
    sim : object
        Forward simulator instance.
    analysis : str, optional
        Analysis flavour. Defaults to the config's ``analysis`` key, so that
        this agrees with :func:`pipt.pipt_init.init_da`, falling back to
        ``"approx"`` if the config does not say. Pass it to override the config.

    Returns
    -------
    object
        The instantiated scheme.
    """
    if analysis is None:
        # The config is the source of truth, so that this agrees with
        # `init_da`. "approx" remains the fallback for a config that does not
        # say -- but a config that *does* say must never be overridden by a
        # default, which is what silently built the wrong scheme before.
        analysis = da_input.get("analysis", "approx")
    return get_scheme(scheme, analysis)(da_input, en_input, sim)


# The algorithm classes themselves. The flavour is a parameter of each, so
# these are plain classes now rather than functions that pick one of eighteen.
from pipt.update_schemes.enkf import EnKF  # noqa: E402
from pipt.update_schemes.enrml import GNEnRML, LMEnRML  # noqa: E402
from pipt.update_schemes.es import ES  # noqa: E402
from pipt.update_schemes.esmda import ESMDA  # noqa: E402

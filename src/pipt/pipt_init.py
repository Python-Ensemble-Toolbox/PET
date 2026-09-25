"""Entry point for constructing an assimilation scheme from parsed config."""

from pipt.update_schemes.registry import get_scheme

__all__ = ["init_da"]


def init_da(da_input, en_input, sim):
    """Build the assimilation scheme object described by the config.

    Parameters
    ----------
    da_input : dict
        Parsed ``dataassim`` section. Must contain ``scheme`` (the algorithm
        name) and ``analysis`` (the flavour).
    en_input : dict
        Parsed ``ensemble`` section.
    sim : object
        Forward simulator instance.

    Returns
    -------
    object
        Instantiated scheme.

    Raises
    ------
    ValueError
        If ``daalg`` is missing or malformed.
    KeyError
        If the requested scheme/analysis combination is not registered. The
        message lists the valid options.
    """
    scheme = da_input.get("scheme")

    if scheme is None:
        if "daalg" in da_input:
            raise ValueError(
                "This config uses the legacy 'daalg' key. It has been replaced "
                "by a single 'scheme' key naming the algorithm:\n\n"
                "    daalg = ['esmda', 'esmda']   ->   scheme = 'esmda'\n\n"
                "Run `pet migrate <config>` to convert the file in place "
                "(the original is kept as <config>.bak)."
            )
        raise ValueError(
            "SCHEME is missing from the data-assimilation config. "
            "It names the assimilation algorithm, e.g. scheme = 'esmda'."
        )

    if not isinstance(scheme, str):
        raise ValueError(
            f"SCHEME must be the algorithm name as a string, e.g. 'esmda'; "
            f"got {scheme!r}."
        )

    analysis = da_input.get("analysis")
    if analysis is None:
        raise ValueError(
            f"ANALYSIS is missing from the data-assimilation config. "
            f"It selects the analysis flavour for scheme '{scheme}'."
        )

    scheme_cls = get_scheme(scheme, analysis)
    return scheme_cls(da_input, en_input, sim)

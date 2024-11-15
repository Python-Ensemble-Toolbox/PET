
from pipt.update_schemes.gies.gies_base import GIESMixIn
from pipt.update_schemes.gies.rlmmac_update import rlmmac_update


class gies_rlmmac(GIESMixIn, rlmmac_update):
    pass
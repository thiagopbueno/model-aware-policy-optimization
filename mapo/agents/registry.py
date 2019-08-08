"""Registry of algorithm names"""


def _import_mapo():
    from mapo.agents import mapo

    return mapo.MAPOTrainer


def _import_off_mapo():
    from mapo.agents import mapo

    return mapo.OffMAPOTrainer


ALGORITHMS = {"MAPO": _import_mapo, "OffMAPO": _import_off_mapo}

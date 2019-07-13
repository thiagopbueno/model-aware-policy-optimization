"""Registry of algorithm names"""


def _import_mapo():
    from mapo.agents import mapo

    return mapo.MAPOTrainer


ALGORITHMS = {"MAPO": _import_mapo}

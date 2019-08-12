"""Registry of algorithm names"""


def _import_mapo():
    from mapo.agents import mapo

    return mapo.MAPOTrainer


def _import_off_mapo():
    from mapo.agents import mapo

    return mapo.OffMAPOTrainer


def _import_td3():
    from mapo.agents import td3

    return td3.TD3Trainer


ALGORITHMS = {"MAPO": _import_mapo, "OffMAPO": _import_off_mapo, "OurTD3": _import_td3}

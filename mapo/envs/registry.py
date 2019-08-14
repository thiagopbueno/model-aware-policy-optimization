"""Registry of custom Gym environment names"""


def _import_navigation_v0(_):
    from mapo.envs import NavigationEnv

    config = {"deceleration_zones": {}}
    return NavigationEnv(**config)


def _import_navigation_v1(_):
    from mapo.envs import NavigationEnv

    config = {}
    return NavigationEnv(**config)


ENVS = {"Navigation-v0": _import_navigation_v0, "Navigation-v1": _import_navigation_v1}

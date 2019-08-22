"""Registry of custom Gym environment names"""


def _import_navigation_v0(_):
    from mapo.envs import TimeAwareTFEnv
    from mapo.envs.navigation import NavigationEnv

    config = {"deceleration_zones": {}}
    return TimeAwareTFEnv(NavigationEnv(**config), horizon=20)


def _import_navigation_v1(_):
    from mapo.envs import TimeAwareTFEnv
    from mapo.envs.navigation import NavigationEnv

    config = {}
    return TimeAwareTFEnv(NavigationEnv(**config), horizon=20)


ENVS = {"Navigation-v0": _import_navigation_v0, "Navigation-v1": _import_navigation_v1}

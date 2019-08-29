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


def _import_navigation1d_v0(_):
    from mapo.envs import TimeAwareTFEnv
    from mapo.envs.navigation1d import Navigation1DEnv

    config = {}
    return TimeAwareTFEnv(Navigation1DEnv(**config), horizon=20)


def _import_navigation1d_v1(_):
    from mapo.envs import TimeAwareTFEnv
    from mapo.envs.navigation1d import Navigation1DEnv

    config = {"n_random_walks": 100}
    return TimeAwareTFEnv(Navigation1DEnv(**config), horizon=20)


ENVS = {
    "Navigation-v0": _import_navigation_v0,
    "Navigation-v1": _import_navigation_v1,
    "Navigation1d-v0": _import_navigation1d_v0,
    "Navigation1d-v1": _import_navigation1d_v1,
}

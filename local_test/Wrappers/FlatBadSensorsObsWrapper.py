import gymnasium as gym

from minigrid.wrappers import FlatObsWrapper, RGBImgObsWrapper

from Wrappers.BadSensorsFlatComponent import BadSensorsFlatComponent


class FlatBadSensorsObsWrapper(gym.ObservationWrapper):
    """
    Este wrappers sirve para mofificar las observaciones de un entorno.
    Estas modificaciones alternan entre:
    - Observación completa del entorno (MDP).
    - Observación parcial del entorno (POMDP).
    """

    def __init__(self, env: gym.Env, seed: int, min_steps: int = 100,
                 max_steps: int = 500, hidden_tiles_min: int = 10,
                 hidden_tiles_max: int = 100, fail_duration: int = 10):
        env = RGBImgObsWrapper(env)
        env = BadSensorsFlatComponent(env, seed, min_steps, max_steps,
                                      hidden_tiles_min, hidden_tiles_max,
                                      fail_duration)
        env = FlatObsWrapper(env)
        super().__init__(env)

    def observation(self, obs):
        return obs

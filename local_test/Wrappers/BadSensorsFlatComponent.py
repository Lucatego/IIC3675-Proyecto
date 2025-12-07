import random
import gymnasium as gym
import numpy as np

from minigrid.wrappers import RGBImgObsWrapper, ReseedWrapper


class BadSensorsFlatComponent(gym.ObservationWrapper):
    """
    Este wrappers sirve para mofificar las observaciones de un entorno.
    Estas modificaciones alternan entre:
    - Observación completa del entorno (MDP).
    - Observación parcial del entorno (POMDP).
    """


    def __init__(self, env: gym.Env, seed: int, min_steps: int = 100,
                 max_steps: int = 500, hidden_tiles_min: int = 10,
                 hidden_tiles_max: int = 100, fail_duration: int = 10):
        env = ReseedWrapper(env, seeds = [seed])
        super().__init__(env)
        env.unwrapped.highlight = False
        self.env.unwrapped.tile_size = 8

        self.min_steps = min_steps
        self.max_steps = max_steps

        self.hidden_tiles_min = hidden_tiles_min
        self.hidden_tiles_max = hidden_tiles_max

        self.fail_duration = fail_duration # Duracion del fallo

        self.steps_failing = 0 # Pasos que llevan fallando
        self.steps = 0 # Pasos que llevan OK
        self.threshold = random.randint(self.min_steps, self.max_steps) # Cada cuantos pasos fallan los sensores

        self.sensors_status_ok = random.choice([True, False])


    def observation(self, obs: dict) -> dict:
        out_obs = obs.copy()

        if not self.sensors_status_ok:
            img = out_obs['image']
            img = self.__apply_partial_observation(img)
            out_obs['image'] = img

        return out_obs


    def __apply_partial_observation(self, img: np.ndarray) -> np.ndarray:
        """
        Pone en negro N tiles aleatorios del grid.
        """
        # Número de tiles a ocultar en cada paso
        N = random.randint(self.hidden_tiles_min, self.hidden_tiles_max)

        tile_pixels = self.env.unwrapped.tile_size
        H, W = self.env.unwrapped.height, self.env.unwrapped.width # dimensiones del grid

        # Todas las posiciones del grid
        ax, ay = self.env.unwrapped.agent_pos
        # print(ax, ay)
        all_tiles = [(x, y) for x in range(W) for y in range(H) if not (x == ax and y == ay)]
        # Elegimos aleatoriamente N tiles
        N = min(N, len(all_tiles))
        tiles_to_hide = random.sample(all_tiles, N)

        # Copiamos la imagen para modificarla
        out = img.copy()

        # Ocultamos cada tile
        for (tx, ty) in tiles_to_hide:
            py = ty * tile_pixels
            px = tx * tile_pixels

            out[py : py + tile_pixels, px : px + tile_pixels] = 0

        return out


    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Si los sensores estan bien
        if self.sensors_status_ok:
            if self.steps >= self.threshold:
                self.__switch_mode()
            self.steps += 1 # Sensores OK
        else: # Si estan mal
            if self.steps_failing >= self.fail_duration:
                self.__switch_mode()
            self.steps_failing += 1 # Sensores fallando

        return self.observation(obs), reward, done, truncated, info


    def __switch_mode(self) -> None:
        # Reinicio de contadores
        self.steps = 0
        self.steps_failing = 0
        # Nueva cantidad de pasos hasta que fallen los sensores al cambiar de Fallando a OK
        if not self.sensors_status_ok:
            self.threshold = random.randint(self.min_steps, self.max_steps)
        # Cambio de modo
        self.sensors_status_ok = not self.sensors_status_ok


    def reset(self, **kwargs):
        self.steps = 0
        self.threshold = random.randint(self.min_steps, self.max_steps)

        obs, info = self.env.reset(**kwargs)

        return self.observation(obs), info

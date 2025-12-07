import random
import gymnasium as gym
import matplotlib.pyplot as plt

from Wrappers.BadSensorsObsWrapper import BadSensorsObsWrapper
from minigrid.manual_control import ManualControl

def getaction(user_input):
    mapping = {
        "a": 0,   # girar izquierda
        "d": 1,   # girar derecha
        "w": 2,   # avanzar
        "p": 3,   # pickup
        "o": 4,   # drop
        "t": 5,   # toggle (abrir puertas)
        "q": 6    # done (terminar)
    }

    if user_input in mapping:
        return mapping[user_input]

    print("Tecla inválida. Usa: w/a/d para moverte, p/o/t para acciones, q para salir.")
    return 6  # por defecto: done

def getactions(user_input):
    actions = []
    for char in user_input:
        action = getaction(char)
        actions.append(action)
    return actions

if __name__ == "__main__":

    gym.envs.registration.register(
        id = 'MiniGrid-Empty-9x9-v0',
        entry_point = 'minigrid.envs:EmptyEnv',
        kwargs = {'size': 9},
        max_episode_steps = 1000
    )

    env = gym.make("MiniGrid-MemoryS17Random-v0",
                   render_mode='rgb_array', 
                   max_steps = 250)
    env = BadSensorsObsWrapper(env = env, seed = None,
                               min_steps = 10, max_steps = 10,
                               hidden_tiles_min = 8, hidden_tiles_max = 32,
                               fail_duration = 5)
    
    obs, _ = env.reset()
    plt.imshow(obs)
    plt.show()
    while True:
        u_input = input()
        action = getaction(u_input)
        obs, reward, done, truncated, info = env.step(action)
        print(reward)
        plt.imshow(obs)
        plt.show()
        if done or truncated: break
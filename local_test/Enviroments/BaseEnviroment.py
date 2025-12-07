from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace

from abc import ABC, abstractmethod

class BaseEnviroment(MiniGridEnv, ABC):
    
    def __init__(self, 
                 size = 8, 
                 agent_start_pos = (1, 1), 
                 agent_start_dir = 0,
                 max_steps: int | None = None, 
                 **kwargs):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func = self._gen_mission)

        if max_steps is None: max_steps = 4 * size ** 2

        super().init(
            mission_space = mission_space,
            grid_size = size,
            # This should be False for a realistic POMDP
            see_through_walls = False,
            max_steps = max_steps,
            **kwargs,
        )

    @abstractmethod
    def _gen_mission():
        pass

    @abstractmethod
    def _gen_grid(self, width, height):
        pass

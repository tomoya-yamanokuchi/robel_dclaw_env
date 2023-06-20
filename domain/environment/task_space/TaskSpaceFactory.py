from .TaskSpaceFactoryNumpy import TaskSpaceFactoryNumpy
from .TaskSpaceFactoryTorch import TaskSpaceFactoryTorch


class TaskSpaceFactory:
    def create(self, env_name: str, mode:str="numpy"):
        if mode=="numpy": return TaskSpaceFactoryNumpy().create(env_name)
        if mode=="torch": return TaskSpaceFactoryTorch().create(env_name)
        raise NotImplementedError()

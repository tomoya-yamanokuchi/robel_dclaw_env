from .TransformerNumpyFactory import TransformerNumpyFactory
from .TransformerTorchFactory import TransformerTorchFactory
from .TaskSpacePositionFactory import TaskSpacePositionFactory
from .TaskSpaceDiffPositionFactory import TaskSpaceDiffPositionFactory


class TaskSpaceFactory:
    @staticmethod
    def create_transformer(env_name, mode):
        if mode == "numpy" : return TransformerNumpyFactory.create(env_name)
        if mode == "torch" : return TransformerTorchFactory.create(env_name)
        raise NotImplementedError()

    @staticmethod
    def create_position(env_name):
        return TaskSpacePositionFactory.create(env_name)

    @staticmethod
    def create_diff_position(env_name):
        return TaskSpaceDiffPositionFactory.create(env_name)

from abc import ABCMeta, abstractmethod



class AbstractEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config):
        pass


    @abstractmethod
    def reset(self):
        pass


    @abstractmethod
    def set_ctrl_task_space(self):
        pass


    @abstractmethod
    def get_state(self):
        pass


    @abstractmethod
    def step(self):
        pass


    @abstractmethod
    def render(self):
        pass

from abc import ABCMeta, abstractmethod


class AbstractTaskSpaceTransformer(metaclass=ABCMeta):
    @abstractmethod
    def task2end(self, task_space_position):
        pass


    @abstractmethod
    def end2task(self, end_effector_position):
        pass

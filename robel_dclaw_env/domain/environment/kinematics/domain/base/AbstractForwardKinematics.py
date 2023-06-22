from abc import ABC, abstractmethod


class AbstractForwardKinematics(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calc(self, theta):
        pass

    @abstractmethod
    def calc_1claw(self, theta):
        pass

    @abstractmethod
    def get_cog(self, theta):
        pass


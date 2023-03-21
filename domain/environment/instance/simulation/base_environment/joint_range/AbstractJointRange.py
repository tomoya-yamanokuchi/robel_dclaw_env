from abc import ABCMeta, abstractmethod



class AbstractJointRange(metaclass=ABCMeta):

    @abstractmethod
    def set_range(self, lower_bound, upper_bound):
        pass

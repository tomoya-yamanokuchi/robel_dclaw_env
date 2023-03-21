from abc import ABCMeta, abstractmethod



class AbstractDynamicsParameter(metaclass=ABCMeta):

    @abstractmethod
    def set(self, randparams_dict: dict) -> None:
        pass

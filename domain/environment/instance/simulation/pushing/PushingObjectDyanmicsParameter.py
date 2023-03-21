import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from ..base_environment.dynamics_parameter.AbstractDynamicsParameter import AbstractDynamicsParameter



class PushingObjectDyanmicsParameter(AbstractDynamicsParameter):
    def __init__(self, sim):
        self.sim = sim


    def set(self, randparams_dict: dict) -> None:
        '''
        set no restriction
        '''
        return 0

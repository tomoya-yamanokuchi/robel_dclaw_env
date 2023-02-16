from .manifold_1d.Manifold1D import Manifold1D
from .end_effector_action_pace.EndEffector2D import EndEffector2D

'''
・環境を生成するクラスです
・新たな独自環境を作成して切り替えたいときには条件分岐を追加することで対応できます
'''

class TaskSpaceFactory:
    def create(self, env_name: str):
        assert type(env_name) == str

        if   env_name == "sim_valve"           : return Manifold1D
        if   env_name == "sim_pushing"         : return EndEffector2D
        # elif env_name == "sim_with_force": return DClawSimulationEnvironmentOptoForce
        # elif env_name == "real"          : return DClawRealEnvironment
        else                             : raise NotImplementedError()


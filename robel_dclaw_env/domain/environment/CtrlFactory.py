
from .instance.simulation.pushing.PushingSetCtrl import PushingCtrl

# from .  .DClawSimulationEnvironment import DClawSimulationEnvironment
# from .simulation.p .DClawSimulationEnvironment import DClawSimulationEnvironment
# from .simulation.DClawSimulationEnvironmentOptoForce import DClawSimulationEnvironmentOptoForce
# from .real.DClawRealEnvironment import DClawRealEnvironment

'''
・環境を生成するクラスです
・新たな独自環境を作成して切り替えたいときには条件分岐を追加することで対応できます
'''

class EnvironmentFactory:
    def create(self, env_name: str):
        assert type(env_name) == str

        if   env_name == "sim"           : return DClawSimulationEnvironment
        # elif env_name == "sim_with_force": return DClawSimulationEnvironmentOptoForce
        # elif env_name == "real"          : return DClawRealEnvironment
        else                             : raise NotImplementedError()


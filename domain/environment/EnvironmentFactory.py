from .instance.simulation.valve.ValveSimulationEnvironment import ValveSimulationEnvironment
from .instance.simulation.pushing.PushingSimulationEnvironment import PushingSimulationEnvironment
# from .simulation.p .DClawSimulationEnvironment import DClawSimulationEnvironment
# from .simulation.DClawSimulationEnvironmentOptoForce import DClawSimulationEnvironmentOptoForce
# from .instance.real.DClawRealEnvironment import DClawRealEnvironment

'''
・環境を生成するクラスです
・新たな独自環境を作成して切り替えたいときには条件分岐を追加することで対応できます
'''

class EnvironmentFactory:
    def create(self, env_name: str):
        assert type(env_name) == str

        # if   env_name == "sim_valve"           : return ValveSimulationEnvironment
        if   env_name == "sim_pushing"         : return PushingSimulationEnvironment
        # elif env_name == "real"          : return DClawRealEnvironment
        raise NotImplementedError()


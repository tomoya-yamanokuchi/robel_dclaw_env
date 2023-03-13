from .instance.simulation.valve.ValveState import ValveState
from .instance.simulation.block_mating.BlockMatingState import BlockMatingState
from .instance.simulation.pushing.PushingState import PushingState

'''
・環境を生成するクラスです
・新たな独自環境を作成して切り替えたいときには条件分岐を追加することで対応できます
'''

class StateFactory:
    def create(self, env_name: str):
        assert type(env_name) == str

        if   env_name == "sim_valve"           : return ValveState
        if   env_name == "sim_block_mating"    : return BlockMatingState
        if   env_name == "sim_pushing"         : return PushingState
        # elif env_name == "sim_with_force": return DClawSimulationEnvironmentOptoForce
        # elif env_name == "real"          : return DClawRealEnvironment
        else                             : raise NotImplementedError()


from .instance.simulation.valve.ValveSimulationEnvironment import ValveSimulationEnvironment
# from .instance.simulation.valve.ValveFeedState import ValveFeedState
from .instance.simulation.valve.ValveState import ValveState


# from .instance.simulation.block_mating.BlockMatingSimulationEnvironment import BlockMatingSimulationEnvironment


from .instance.simulation.pushing.PushingSimulationEnvironment import PushingSimulationEnvironment
from .instance.simulation.pushing.PushingFeedState import PushingFeedState


'''
・環境を生成するクラスです
・新たな独自環境を作成して切り替えたいときには条件分岐を追加することで対応できます
'''

class EnvironmentFactory:
    def create(self, env_name: str):
        assert type(env_name) == str

        if   env_name == "sim_valve"           : return (ValveSimulationEnvironment, ValveState)
        # if   env_name == "sim_block_mating"    : return BlockMatingSimulationEnvironment
        if   env_name == "sim_pushing"         : return (PushingSimulationEnvironment, PushingFeedState)
        # elif env_name == "real"          : return DClawRealEnvironment
        raise NotImplementedError()


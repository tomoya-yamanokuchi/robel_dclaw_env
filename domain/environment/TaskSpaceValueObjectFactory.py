from .task_space.manifold_1d_torch.Manifold1D import Manifold1D
# from .task_space.end_effector_2d_torch


'''
・環境を生成するクラスです
・新たな独自環境を作成して切り替えたいときには条件分岐を追加することで対応できます
'''

class TaskSpaceValueObjectFactory:
    def create(self, env_name: str):
        assert type(env_name) == str

        if   env_name == "sim_valve"           : return Manifold1D
        if   env_name == "sim_pushing"         : return
        # elif env_name == "real"          : return DClawRealEnvironment
        raise NotImplementedError()


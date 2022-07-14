from .DClawEnvironment import DClawEnvironment
from .Force import Force

'''
・環境を生成するクラスです
・新たな独自環境を作成して切り替えたいときには条件分岐を追加することで対応できます
'''

class EnvironmentFactory:
    def create(self, env_name: str):
        assert type(env_name) == str

        if   env_name == "dclaw": return DClawEnvironment
        elif env_name == "force" : return Force
        # elif env_name == "new_env" : return NewEnv (このような形で追加できます)
        else                  : raise NotImplementedError()


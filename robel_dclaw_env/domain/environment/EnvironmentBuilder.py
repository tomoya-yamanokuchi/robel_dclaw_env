from .EnvironmentFactory import EnvironmentFactory


class EnvironmentBuilder:
    def build(self, config_env):
        env_sub, state_sub = EnvironmentFactory().create(env_name=config_env.env.env_name)
        env                = env_sub(config_env.env)
        init_state         = state_sub(**config_env.env.init_state)
        return env, init_state

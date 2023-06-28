import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.EnvironmentFactory import EnvironmentFactory
from robel_dclaw_env.domain.forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from robel_dclaw_env.custom_service import time_as_string



class DataCollection:
    def __init__(self, config):
        self.config = config


    def run_forward_model(self, rollout_function, ctrl, TaskSpaceAbs, TaskSpaceDiff, dataset_name):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=self.config.env.env_name)
        init_state = state_subclass(**self.config.env.init_state)
        multiproc          = ForwardModelMultiprocessing(verbose=False)
        results, proc_time = multiproc.run(
            rollout_function = rollout_function,
            constant_setting = {
                "env_subclass" : env_subclass,
                "config"       : self.config,
                "init_state"   : init_state,
                "TaskSpaceAbs" : TaskSpaceAbs,
                "TaskSpaceDiff": TaskSpaceDiff,
                "dataset_name" : dataset_name + "_" + time_as_string(),
            },
            ctrl = ctrl,
        )




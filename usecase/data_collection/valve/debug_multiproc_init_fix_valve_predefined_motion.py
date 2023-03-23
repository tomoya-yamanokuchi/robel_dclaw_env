import os
import copy
import pprint
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from usecase.data_collection.rollout.rollout_dataset_collection_random_motion import rollout_dataset_collection
from usecase.data_collection.rollout.rollout_dataset_collection_debug import rollout_dataset_collection_debug
from ctrl_set.six_predefined_ctrl_set import SixPredefinedCtrlSet
from custom_service import time_as_string, NTD


from domain.environment.instance.simulation.valve.ValveReturnState import ValveReturnState
from domain.environment.instance.simulation.valve.ValveReturnCtrl  import ValveReturnCtrl
from multiprocessing import Queue, JoinableQueue


class DataCollection:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        step        = config.run.step
        cumsum_ctrl = SixPredefinedCtrlSet().get(step)
        cumsum_ctrl = cumsum_ctrl[:1]
        ctrl        = NTD(init_state.task_space_position) + cumsum_ctrl

        # import ipdb; ipdb.set_trace()

        constant_setting = {
            "env_subclass" : env_subclass,
            "config"       : config,
            "init_state"   : init_state,
            "dataset_name" : time_as_string(),
            "domain_index" : 1,
            "ReturnState"  : ValveReturnState,
            "ReturnCtrl"   : ValveReturnCtrl,
        }


        rollout_dataset_collection_debug(constant_setting, ctrl)



if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = DataCollection()
        demo.run(config)

    main()

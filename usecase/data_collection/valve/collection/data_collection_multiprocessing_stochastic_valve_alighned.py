import os
import copy
import pprint
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.__StateFactory import StateFactory
# from domain.environment.instance.simulation.DClawCtrl import DClawCtrl as CtrlState
# from domain.environment.instance.simulation.simulation.base_environment.ImageObs import ImageObs
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository

from domain.environment.multiprocessing.EnvironmentMultiprocessing_develop import EnvironmentMultiprocessing
# from domain.environment.multiprocessing.EnvironmentMultiprocessing import EnvironmentMultiprocessing

from domain.environment.multiprocessing.EnvironmentConstantSetting import EnvironmentConstantSetting








class Demo_task_space:
    def run(self, config):
        config.env.camera.z_distance = 0.4 # ロボットがフレームアウトする情報欠損を起こさないようにカメラを引きで設定
        env_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        task_space_position_init = np.array(
            [
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
                [0.14, 0.14, 0.14], # 指右上寄せ
            ]
        )
        num_action_variation = task_space_position_init.shape[0]
        init_state   = EnvState(
            robot_position        = np.tile(np.array(config.env.robot_position_init).reshape(1, -1), (num_action_variation, 1)),
            robot_velocity        = np.tile(np.array(config.env.robot_velocity_init).reshape(1, -1), (num_action_variation, 1)),
            object_position       = np.tile(np.array(config.env.object_position_init).reshape(1, -1), (num_action_variation, 1)),
            object_velocity       = np.tile(np.array(config.env.object_velocity_init).reshape(1, -1), (num_action_variation, 1)),
            end_effector_position = None,
            task_space_positioin  = task_space_position_init,
            mode  = "sequence",
        )
        # import ipdb; ipdb.set_trace()

        step = config.run.step
        dim_task_space = 3

        # -----------------------------------------------------------------------------
        #                                    ctrl
        # -----------------------------------------------------------------------------
        const = 0.05
        # import ipdb; ipdb.set_trace()
        ctrl_task_diff = np.stack(
            (
                np.stack((np.zeros(step) + const, np.zeros(step),           np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step) - const, np.zeros(step),           np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step) + const,   np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step) - const,   np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step),           np.zeros(step) + const) , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step),           np.zeros(step) - const) , axis=-1),
            ), axis=0
        )
        num_ctrl = ctrl_task_diff.shape[0]

        # init_state =
        # import ipdb; ipdb.set_trace()

        chunked_input = []
        for i in range(config.run.sequence):
            chunked_input_unit_dict = {
                "ctrl_task_diff" : ctrl_task_diff,
                "init_state"     : init_state,
            }
            chunked_input.append(chunked_input_unit_dict)
        # import ipdb; ipdb.set_trace()
        # ----------------------------------------------------------------------------------
        import datetime
        dt_now       = datetime.datetime.now()
        dataset_name = "dataset_{}{}{}{}{}{}".format(
            dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute, dt_now.second
        )
        print(dataset_name)

        multiproc        = EnvironmentMultiprocessing()
        constant_setting = EnvironmentConstantSetting(
            env_subclass = env_subclass,
            config       = config,
            # init_state   = init_state,
            dataset_name = dataset_name,
        )

        result_list, proc_time = multiproc.run_from_chunked_ctrl(
            function         = rollout,
            constant_setting = constant_setting,
            chunked_input    = chunked_input,
        )
        print("-------------------------------")
        print("   procces time : {: .3f} [sec]".format(proc_time))
        print("    result_list : {}".format(result_list))
        print("-------------------------------")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()

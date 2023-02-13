import os
import copy
import pprint
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState
from domain.environment.DClawCtrl import DClawCtrl as CtrlState
from domain.environment.simulation.base_environment.ImageObs import ImageObs
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository

from domain.environment.multiprocessing.EnvironmentMultiprocessing_develop import EnvironmentMultiprocessing
# from domain.environment.multiprocessing.EnvironmentMultiprocessing import EnvironmentMultiprocessing

from domain.environment.multiprocessing.EnvironmentConstantSetting import EnvironmentConstantSetting




def rollout(constant_setting, queue_input, queue_result):
    '''
    マルチプロセッシングで実行する関数
    '''
    ctrl_index, chunked_ctrl_input = queue_input.get() # キューからインデックスと制御入力を取り出す

    # print(type(chunked_ctrl_input))
    # chunked_ctrl_task_diff = chunked_ctrl_input

    # print(type(chunked_ctrl_input[0]))
    # print(type(chunked_ctrl_input[1]))
    # print()

    num_chunked_ctrl = len(chunked_ctrl_input)
    assert type(ctrl_index) == int
    # print("ctrl_index ----> ", ctrl_index)

    # 開始までの待ち時間をランダムに決定
    np.random.seed(ctrl_index)
    wait_time = np.random.rand()*5
    print(wait_time)
    time.sleep(wait_time)

    env_subclass = constant_setting.env_subclass
    config       = constant_setting.config
    # init_state   = constant_setting.init_state
    dataset_name = constant_setting.dataset_name

    env          = env_subclass(config.env)
    repository   = Repository(dataset_dir="./dataset", dataset_name=dataset_name, read_only=False)
    for batch_index, chunked_ctrl_input_dict in enumerate(chunked_ctrl_input):

        ctrl_task_diff = chunked_ctrl_input_dict["ctrl_task_diff"]
        init_state     = chunked_ctrl_input_dict["init_state"]

        num_batch, step, dim = ctrl_task_diff.shape
        assert dim == 3
        env.randomize_texture_mode = "per_reset" # (1) テクスチャをバッチ単位で変更するためper_resetに設定
        env.reset(init_state.get_step(0))        # (2) resetによってテクスチャをランダム化
        env.randomize_texture_mode = "static"    # (3) バッチ内ではテクスチャを固定させるためstaticに設定
        for n in range(num_batch):
            repository.open(filename='domain{}-{}_action{}'.format(ctrl_index, batch_index, n))

            env.reset(init_state.get_step(n))
            image_list = []
            state_list = []
            ctrl_list  = []
            for t in range(step):
                img   = env.render()
                state = env.get_state()
                ctrl  = env.set_ctrl_task_diff(ctrl_task_diff[n, t])

                image_list.append(img)
                state_list.append(state)
                ctrl_list.append(ctrl)

                # env.view()
                env.step()

            repository.assign("image", image_list, ImageObs)
            repository.assign("state", state_list, EnvState)
            repository.assign("ctrl",  ctrl_list, CtrlState)
            repository.close()
            print("[index {}-({}/{})] sequence {}/{}".format(ctrl_index, batch_index+1, num_chunked_ctrl, n+1, num_batch))
    queue_result.put((ctrl_index, [ctrl_index])) # 結果とバッチインデックスをキューに入れる
    queue_input.task_done() # キューを終了する



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

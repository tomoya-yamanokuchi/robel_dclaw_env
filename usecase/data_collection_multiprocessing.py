import os
import copy
import pprint
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState
from domain.environment.DClawCtrl import DClawCtrl as CtrlState
from domain.environment.ImageObs import ImageObs
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
from domain.environment.multiprocessing.EnvironmentMultiprocessing import EnvironmentMultiprocessing
from domain.environment.multiprocessing.EnvironmentConstantSetting import EnvironmentConstantSetting



def rollout(constant_setting, queue_input, queue_result):
    '''
    マルチプロセッシングで実行する関数
    '''
    ctrl_index, chunked_ctrl_task_diff = queue_input.get() # キューからインデックスと制御入力を取り出す
    num_chunked_ctrl = len(chunked_ctrl_task_diff)
    assert type(ctrl_index) == int
    # print("ctrl_index ----> ", ctrl_index)

    # 開始までの待ち時間をランダムに決定
    np.random.seed(ctrl_index)
    wait_time = np.random.rand()*5
    print(wait_time)
    time.sleep(wait_time)

    env_subclass = constant_setting.env_subclass
    config       = constant_setting.config
    init_state   = constant_setting.init_state
    dataset_name = constant_setting.dataset_name

    env          = env_subclass(config.env)
    repository   = Repository(dataset_dir="./dataset", dataset_name=dataset_name, read_only=False)
    for batch_index, ctrl_task_diff in enumerate(chunked_ctrl_task_diff):
        num_batch, step, dim = ctrl_task_diff.shape
        assert dim == 3
        env.randomize_texture_mode = "per_step" # テクスチャをバッチ単位で変更するためここで変える
        env.reset(init_state)
        # env.randomize_texture_mode = "static" # バッチ内では固定させる
        for n in range(num_batch):
            repository.open(filename='domain{}-{}_action{}'.format(ctrl_index, batch_index, n))
            env.reset(init_state)
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
        env_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state   = EnvState(
            robot_position        = np.array(config.env.robot_position_init),
            robot_velocity        = np.array(config.env.robot_velocity_init),
            object_position       = np.array(config.env.object_position_init),
            object_velocity       = np.array(config.env.object_velocity_init),
            end_effector_position = None,
            task_space_positioin  = np.array(config.env.task_space_position_init),
        )

        step = config.run.step
        dim_task_space = 3

        # ctrl ----------------------------------------------------------------------------
        const = 0.05
        ctrl_task_diff_idx0_plus = np.stack(
            (np.zeros(step) + const,
             np.zeros(step),
             np.zeros(step),
            ), axis=-1
        )
        ctrl_task_diff_idx1_plus = copy.deepcopy(ctrl_task_diff_idx0_plus[:, [1, 0, 2]])
        ctrl_task_diff_idx2_plus = copy.deepcopy(ctrl_task_diff_idx0_plus[:, [2, 1, 0]])

        ctrl_task_diff_idx0_minus = np.stack(
            (np.zeros(step) - const,
             np.zeros(step),
             np.zeros(step),
            ), axis=-1
        )
        ctrl_task_diff_idx1_minus = copy.deepcopy(ctrl_task_diff_idx0_minus[:, [1, 0, 2]])
        ctrl_task_diff_idx2_minus = copy.deepcopy(ctrl_task_diff_idx0_minus[:, [2, 1, 0]])

        ctrl_task_diff_plus       = np.zeros([step, dim_task_space]) + const
        ctrl_task_diff_minus      = np.zeros([step, dim_task_space]) - const

        ctrl_task_diff = np.stack(
            (
                ctrl_task_diff_idx0_plus,
                ctrl_task_diff_idx1_plus,
                ctrl_task_diff_idx2_plus,
                ctrl_task_diff_idx0_minus,
                ctrl_task_diff_idx1_minus,
                ctrl_task_diff_idx2_minus,
                ctrl_task_diff_plus,
                ctrl_task_diff_minus,
            ), axis=0
        )
        num_ctrl = ctrl_task_diff.shape[0]


        chunked_ctrl = []
        for i in range(config.run.sequence):
            chunked_ctrl.append(ctrl_task_diff)
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
            init_state   = init_state,
            dataset_name = dataset_name,
        )

        result_list, proc_time = multiproc.run_from_chunked_ctrl(
            function         = rollout,
            constant_setting = constant_setting,
            chunked_ctrl     = chunked_ctrl,
        )
        print("-------------------------------")
        print("   procces time : {: .3f} [sec]".format(proc_time))
        print("    result_list : {}".format(result_list))
        print("-------------------------------")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()
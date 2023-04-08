import copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from domain.environment.DClawState import DClawState as EnvState
from domain.environment.multiprocessing.EnvironmentMultiprocessing import EnvironmentMultiprocessing
from domain.environment.multiprocessing.EnvironmentConstantSetting import EnvironmentConstantSetting



def rollout(constant_setting, queue_input, queue_result):
    '''
    マルチプロセッシングで実行する関数を定義
    '''
    ctrl_index, ctrl_task_diff = queue_input.get() # キューからインデックスと制御入力を取り出す
    assert type(ctrl_index) == int

    num_batch, step, dim = ctrl_task_diff.shape
    assert dim == 3

    env_subclass = constant_setting.env_subclass
    config       = constant_setting.config
    init_state   = constant_setting.init_state

    env          = env_subclass(config.env)
    reward       = np.zeros(num_batch)
    for n in range(num_batch):
        env.reset(init_state)
        env.randomize_texture()
        for t in range(step):
            img   = env.render()
            state = env.get_state()
            env.set_ctrl_task_diff(ctrl_task_diff[n, t])
            env.view()
            env.step()
        reward[n] = copy.deepcopy(env.get_state().object_position) # reward をバルブの回転角度として取り出す
    queue_result.put((ctrl_index, reward)) # 結果とインデックスをキューに入れる
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

        dim_task_space = 3 # 1本の指につき1次元の拘束をするので合計3次元
        ctrl_task_diff = np.random.uniform(low=-0.03, high =0.03, size=(config.run.sequence, dim_task_space))
        ctrl_task_diff = np.tile(ctrl_task_diff[:, np.newaxis, :], (1, config.run.step, 1))

        multiproc        = EnvironmentMultiprocessing()
        constant_setting = EnvironmentConstantSetting(
            env_subclass = env_subclass,
            config       = config,
            init_state   = init_state,
        )

        result_list, proc_time = multiproc.run(
            function         = rollout,
            constant_setting = constant_setting,
            ctrl             = ctrl_task_diff,
        )
        print("-------------------------------")
        print("   procces time : {: .3f} [sec]".format(proc_time))
        print("     min reward : {}".format(np.sort(np.array(result_list))[0]))
        print("     max reward : {}".format(np.sort(np.array(result_list))[-1]))
        print("-------------------------------")


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()
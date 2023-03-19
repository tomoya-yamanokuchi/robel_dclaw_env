import cv2, time, copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
# from domain.environment.__StateFactory import StateFactory
from custom_service import print_info


class Demo_task_space:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        # state = StateFactory().create(env_name=config.env.env_name)

        env = env_subclass(config.env)
        init_state = state_subclass(
            task_space_positioin = np.array(config.env.task_space_position_init),
            robot_velocity       = np.array(config.env.robot_velocity_init),
            object_position      = np.array(config.env.object_position_init),
            object_velocity      = np.array(config.env.object_velocity_init),
        )
        # import ipdb; ipdb.set_trace()

        step           = 100
        dim_task_space = 6
        ctrl_task_diff = np.zeros([step, dim_task_space]) # 範囲:[0, 1]

        ctrl_task_diff[:, ::2] += np.zeros([step, 3]) + 0.00
        for s in range(10):
            time_start = time.time()
            env.reset(init_state)
            print("\n*** reset ***\n")
            task_t = env.get_state().task_space_positioin
            task_g = copy.deepcopy(task_t)

            for i in range(25):
                img   = env.render()
                state = env.get_state()
                env.view()

                task_g[0] += 0.00
                task_g[1] += 0.1

                env.set_ctrl_task_space(task_g)
                env.step(is_view=False)

                # import ipdb; ipdb.set_trace()
                # print("body_inertia = ", env.sim.data.body_inertia[21])

            time_end = time.time()
            print("time epoch = ", time_end - time_start)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config)

    main()

import cv2, time, copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory



class Demo_task_space:
    def run(self, config, task_space_positioin):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        step, dim_ctrl = task_space_positioin.shape
        # assert num_data == 1
        # import ipdb; ipdb.set_trace()

        env = env_subclass(config.env, use_render=True)

        for s in range(10):
            time_start = time.time()
            env.reset(init_state)
            env.render()
            for t in range(step):
                # img   = env.render()
                # state = env.get_state()

                env.set_ctrl_task_space(task_space_positioin[t])
                # env.view()
                env.step(is_view=True)

            time_end = time.time()
            print("time epoch = ", time_end - time_start)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
    def main(config: DictConfig):

        task_space_positioin = np.load(
            "./best_elite_action/" + \
            "best_elite_action-[num_cem_iter=1]-[planning_horizon=10]-[num_sample=50]-2023330184158.npy"
        )

        config.env.viewer.is_Offscreen = False

        demo = Demo_task_space()
        demo.run(config, task_space_positioin)

    main()

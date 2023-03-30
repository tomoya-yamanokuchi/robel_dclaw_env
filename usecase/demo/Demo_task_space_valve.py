import cv2, time, copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
from custom_service import print_info



class Demo_task_space:
    def run(self, config):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        env        = env_subclass(config.env)
        init_state = state_subclass(**config.env.init_state)

        step           = 100
        dim_task_space = 3

        for s in range(10):
            time_start = time.time()
            env.reset(init_state); print("\n*** reset ***\n")
            state  = env.get_state()
            task_t = state.state['task_space_position'].value.squeeze()
            task_g = copy.deepcopy(task_t)
            # task_g -= 0.2

            for t in range(step):
                img   = env.render()
                state = env.get_state()
                env.view()

                # task_g[0] += 0.00
                # task_g[1] += 0.1
                task_g -= 0.05
                # import ipdb; ipdb.set_trace()

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

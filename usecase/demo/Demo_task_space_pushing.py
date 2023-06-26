import cv2, time, copy
import numpy as np
from robel_dclaw_env.custom_service import print_info, NTD
from robel_dclaw_env.domain import EnvironmentBuilder


class Demo_task_space:
    def run(self, config):
        env_struct        = EnvironmentBuilder().build(config, mode="numpy")
        env               = env_struct["env"]
        init_state        = env_struct["init_state"]
        TaskSpacePosition = env_struct["TaskSpacePosition"]

        step           = 100
        dim_task_space = 6

        for s in range(10):
            time_start = time.time()
            env.reset(init_state); print("\n*** reset ***\n")
            env.render()
            state  = env.get_state()
            task_t = state.collection['task_space_position'].value.squeeze()
            task_g = copy.deepcopy(task_t)

            for t in range(step):
                # img   = env.render()
                state = env.get_state()
                env.view()

                # import ipdb; ipdb.set_trace()
                task_g[0] += 0.1
                task_g[1] += -0.1

                env.set_ctrl_task_space(TaskSpacePosition(NTD(task_g)))
                env.step(is_view=True)

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

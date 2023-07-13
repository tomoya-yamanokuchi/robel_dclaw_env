import cv2, time, copy
import numpy as np
from robel_dclaw_env.custom_service import NTD
from robel_dclaw_env.domain import EnvironmentBuilder


class Demo_task_space:
    def run(self, config, episode, step):
        env_struct        = EnvironmentBuilder().build(config, mode="numpy")
        env               = env_struct["env"]
        init_state        = env_struct["init_state"]
        TaskSpacePosition = env_struct["TaskSpacePosition"]

        task_space_position_init = init_state.collection['task_space_position'].value.squeeze()

        sac = SAC(config_sac)

        for m in range(episode):
            env.reset(init_state, verbose=True)
            task_g = copy.deepcopy(task_space_position_init)
            for t in range(step):
                # img   = env.render()    # img.collection["canonical"]
                state = env.get_state() # state.collection["robot_position"]

                robot_position  = state.collection["robot_position"]
                object_position = state.collection["object_position"]

                action = sac.action(np.concatenate([robot_position, object_position]))

                env.set_ctrl_task_space(TaskSpacePosition(NTD(action)))
                env.step(is_view=False)

                reward = env.get_reward()


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="/nfs/monorepo_ral2023/robel_dclaw_env/conf", config_name="config.yaml")
    def main(config: DictConfig):
        demo = Demo_task_space()
        demo.run(config, episode=5, step=100)

    main()

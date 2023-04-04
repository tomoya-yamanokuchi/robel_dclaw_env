import os, cv2, time, copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.EnvironmentFactory import EnvironmentFactory
import pickle
from visualize.ReplayVisualization import ReplayVisualization
from domain.reference.ValveReference import ValveReference
from domain.icem_mpc.TrajectoryVisualization import TrajectoryVisualization
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold as TaskSpace
from custom_service import NTD


class Demo_task_space:
    def run(self, config, task_space_differential_position):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        step, dim_ctrl = task_space_differential_position.shape

        replay_task_space_abs_position = []
        replay_object_position         = []
        env       = env_subclass(config.env, use_render=False)
        reference = ValveReference(step)
        for s in range(1):
            time_start = time.time()
            env.reset(init_state)
            # env.render()
            for t in range(step):
                # img   = env.render()
                state = env.get_state(); replay_object_position.append(state.state["object_position"].value)
                # -----
                task_space_position = state.state["task_space_position"]
                task_space_ctrl     = task_space_position + TaskSpace(NTD(task_space_differential_position[t]))
                # import ipdb; ipdb.set_trace()
                # actions             = TaskSpace(action_doi.construct(cumsum_actions)).value
                env.set_ctrl_task_space(task_space_ctrl)
                replay_task_space_abs_position.append(task_space_ctrl.value.squeeze())
                # -----
                env.step(is_view=False)
            replay_object_position.append(state.state["object_position"].value)

            time_end = time.time()
            print("time epoch = ", time_end - time_start)

        return np.stack(replay_task_space_abs_position), np.stack(replay_object_position)


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig
    from custom_service import load_best_elite_sequence
    from visualize.ReplayActionVisualization import ReplayActionVisualization
    from visualize.ReplayObjectPpositionVisualization import ReplayObjectPpositionVisualization


    @hydra.main(version_base=None, config_path="../../conf", config_name="config.yaml")
    def main(config: DictConfig):

        result_best_elite_sequence = load_best_elite_sequence(
            load_path = "./best_elite_sequence/" + \
            "best_elite_sequence-[num_cem_iter=7]-[planning_horizon=10]-[num_sample=700]-2023441945.pkl",
        )

        config.env.viewer.is_Offscreen = False

        demo = Demo_task_space()
        replay_task_space_abs_position, replay_object_state_sequence = demo.run(config,
            result_best_elite_sequence["task_space_differential_position"]
        )

        vis_action = ReplayActionVisualization(
            dim_action = result_best_elite_sequence["task_space_differential_position"].shape[-1],
            figsize    = (5, 7),
        )
        vis_action.plot(result_best_elite_sequence["task_space_abs_position"])
        vis_action.plot(replay_task_space_abs_position)
        vis_action.save("./replay_task_space_abs_position.png")


        vis_object = ReplayObjectPpositionVisualization(
            dim_action = result_best_elite_sequence["best_object_state_sequence"].shape[-1],
            figsize    = (5, 5),
        )
        vis_object.plot(result_best_elite_sequence["best_object_state_sequence"])
        vis_object.plot(replay_object_state_sequence[1:])
        vis_object.save("./replay_object_position.png")


    main()

import os, cv2, time, copy
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env'); sys.path.insert(0, './cdsvae')
from domain.environment.EnvironmentFactory import EnvironmentFactory
import pickle
from visualize.ReplayVisualization import ReplayVisualization
from domain.reference.ValveReference import ValveReference
from domain.icem_mpc.icem_mpc.visualization.elements.utils.TrajectoryVisualization import TrajectoryVisualization
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold as TaskSpace
from custom_service import NTD, concat


class Demo_task_space:
    def run(self, config, task_space_differential_position):
        env_subclass, state_subclass = EnvironmentFactory().create(env_name=config.env.env_name)
        init_state = state_subclass(**config.env.init_state)

        step, dim_ctrl = task_space_differential_position.shape

        replay_task_space_abs_position = []
        replay_object_position         = []
        joint_space_position           = []

        env = env_subclass(config.env, use_render=True)
        for s in range(1):
            time_start = time.time()
            env.reset(init_state)
            env.render()
            for t in range(step):
                # img   = env.render()
                state = env.get_state(); replay_object_position.append(state.collection["object_position"].value)
                # -----
                task_space_position = state.collection["task_space_position"]
                task_space_ctrl     = task_space_position + TaskSpace(NTD(task_space_differential_position[t]))
                # import ipdb; ipdb.set_trace()
                # actions             = TaskSpace(action_doi.construct(cumsum_actions)).value
                ctrl = env.set_ctrl_task_space(task_space_ctrl);



                joint_space_position.append(ctrl.collection["joint_space_position"].value.squeeze())


                replay_task_space_abs_position.append(task_space_ctrl.value.squeeze())
                # -----
                env.step(is_view=True)
            replay_object_position.append(state.collection["object_position"].value)
            joint_space_position.append(ctrl.collection["joint_space_position"].value.squeeze())

            time_end = time.time()
            print("time epoch = ", time_end - time_start)

        return np.stack(replay_task_space_abs_position), np.stack(replay_object_position), np.stack(joint_space_position)


if __name__ == "__main__":
    import hydra
    from omegaconf import OmegaConf, DictConfig
    from custom_service import load_best_elite_sequence
    from visualize.ReplayActionVisualization import ReplayActionVisualization
    from visualize.ReplayObjectPpositionVisualization import ReplayObjectPpositionVisualization
    from domain.icem_mpc.icem_repository.iCEM_Repository import iCEM_Repository
    from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
    from cdsvae.custom import PlotLineMultiDim

    @hydra.main(version_base=None, config_path="../../conf", config_name="config_test.yaml")
    def main(config: DictConfig):
        repository = Repository(**config.nominal, read_only= True)
        repository.open(filename="nominal")
        ctrl = copy.deepcopy(repository.repository["ctrl"])
        repository.close()

        config.env.viewer.is_Offscreen = False

        demo = Demo_task_space()
        _, replay_object_state_sequence, joint_space_position = demo.run(config, ctrl["task_space_diff_position"])


        # --------- valve abs position ---------------
        step       = replay_object_state_sequence.shape[0] - 1
        reference  = ValveReference(step)
        vis_object = ReplayObjectPpositionVisualization(
            dim_action = replay_object_state_sequence.shape[-1],
            figsize    = (5, 5),
        )
        vis_object.plot_target(np.linspace(1, step, step), reference.get_as_radian(current_step=0).squeeze())
        vis_object.plot_path(  np.linspace(0, step+1, step+1), replay_object_state_sequence)
        vis_object.save(os.path.join(config.nominal.dataset_dir,config.nominal.dataset_name, "replay_object_position.png"))


        # --------- cos, sin of valve state ---------------
        plotmulti = PlotLineMultiDim(dim=2,  figsize    = (5, 5))
        plotmulti.plot(reference.get_as_polar_coordinates_cycle_120degree(current_step=0).squeeze())
        plotmulti.save_fig(os.path.join(config.nominal.dataset_dir,config.nominal.dataset_name, "replay_object_position_cos_sin.png"))


        # --------- joint_space_position ctrl ---------------
        joint_space_position_model_use = np.take(joint_space_position, [0, 1, 3,4, 6,7], axis=-1)
        plotmulti = PlotLineMultiDim(
            dim        = joint_space_position_model_use.shape[-1],
            figsize    = (5, 5),
        )
        plotmulti.plot(joint_space_position_model_use)
        plotmulti.save_fig(os.path.join(config.nominal.dataset_dir,config.nominal.dataset_name, "replay_joint_space_position_model_use.png"))


    main()

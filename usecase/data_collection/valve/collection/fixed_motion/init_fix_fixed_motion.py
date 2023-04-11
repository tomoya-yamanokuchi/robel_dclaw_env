import os
import copy
import pprint
import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.data_collection.DataCollection import DataCollection
from usecase.data_collection.rollout.rollout_dataset_collection_with_differential_ctrl import rollout_dataset_collection_with_differential_ctrl
from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold
from domain.environment.task_space.manifold_1d.TaskSpaceDifferentialPositionValue_1D_Manifold import TaskSpaceDifferentialPositionValue_1D_Manifold
from custom_service import time_as_string


def run(config, ctrl):
    data_collection = DataCollection(config)
    data_collection.run_forward_model(
        rollout_function = rollout_dataset_collection_with_differential_ctrl,
        ctrl             = ctrl,
        TaskSpaceAbs     = TaskSpacePositionValue_1D_Manifold,
        TaskSpaceDiff    = TaskSpaceDifferentialPositionValue_1D_Manifold,
        dataset_name     = "fixed_motion_N{}".format(ctrl.shape[0]) + "_" + time_as_string(),
    )


if __name__ == "__main__":
    import hydra
    from omegaconf import DictConfig

    @hydra.main(version_base=None, config_path="../../../../../conf", config_name="config.yaml")
    def main(config: DictConfig):

        config.env.camera.z_distance = 0.4 # ロボットがフレームアウトする情報欠損を起こさないようにカメラを引きで設定
        config.env.init_state.task_space_position = [0.14, 0.14, 0.14] # 指の初期位置を中心に？

        step  = config.run.step
        const = 0.05

        task_space_diff_position = np.stack(
            (
                np.stack((np.zeros(step) + const, np.zeros(step),           np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step) - const, np.zeros(step),           np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step) + const,   np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step) - const,   np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step),           np.zeros(step) + const) , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step),           np.zeros(step) - const) , axis=-1),
            ), axis=0
        )

        task_space_diff_position = np.tile(task_space_diff_position, (config.run.sequence, 1, 1))

        run(config, task_space_diff_position)

    main()

import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository



repository = Repository(
    dataset_dir  = "./dataset",
    dataset_name = "random_action_all_claw_NumSample600_NumColoredNoiseExponent6_20234944611",
    read_only    = True
)


fnames = repository.get_filenames()


for f in fnames:
    # print(f)
    repository.open(f)

    img_can                  = repository.get_image(key="canonical")
    object_position          = repository.get_state(key="object_position")
    robot_position           = repository.get_state(key="robot_position")
    task_space_diff_position = repository.get_ctrl(key="task_space_diff_position")
    joint_space_position     = repository.get_ctrl(key="joint_space_position")

    print(img_can.shape, object_position.shape, robot_position.shape, task_space_diff_position.shape, joint_space_position.shape)
    repository.close()


# import ipdb; ipdb.set_trace()

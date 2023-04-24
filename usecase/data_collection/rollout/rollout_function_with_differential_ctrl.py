import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.forward_model_multiprocessing.ForkedPdb import ForkedPdb
from custom_service import wait_time, NTD




def rollout_function_with_differential_ctrl(constant_setting, queue_input, queue_result):
    index_chunk, task_space_differential_position = queue_input.get()
    num_chunk, step, dim_ctrl = task_space_differential_position.shape
    wait_time(const=5, seed=index_chunk)

    env_subclass  = constant_setting["env_subclass"]
    config        = constant_setting["config"]
    init_state    = constant_setting["init_state"]
    TaskSpaceDiff = constant_setting["TaskSpaceDiff"]

    env = env_subclass(config.env, use_render=False)

    # << ------ rollout ------- >>
    robot_state_trajectory  = []
    object_state_trajectory = []
    for n in range(num_chunk):
        robot_position_1seq = []
        object_state_1seq   = []
        env.reset(init_state)
        for t in range(step):
            state = env.get_state()
            robot_position_1seq.append(state.collection["robot_position"].value)
            object_state_1seq.append(state.collection["object_position"].value)
            # -----
            task_space_position = state.collection["task_space_position"]
            task_space_ctrl     = task_space_position + TaskSpaceDiff(NTD(task_space_differential_position[n,t]))
            env.set_ctrl_task_space(task_space_ctrl)
            # -----
            env.step()
        state = env.get_state()
        robot_position_1seq.append(state.collection["robot_position"].value)
        object_state_1seq.append(state.collection["object_position"].value)
        # -----
        robot_state_trajectory.append(np.stack(robot_position_1seq))
        object_state_trajectory.append(np.stack(object_state_1seq))
    robot_state_trajectory  = np.stack(robot_state_trajectory)
    object_state_trajectory = np.stack(object_state_trajectory)

    # << ---- queue procedure ----- >>
    queue_result.put({
        "index_chunk"            : index_chunk,
        "robot_state_trajectory" : robot_state_trajectory,
        "object_state_trajectory": object_state_trajectory,
    })
    queue_input.task_done()

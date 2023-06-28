import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.forward_model_multiprocessing.ForkedPdb import ForkedPdb
from robel_dclaw_env.custom_service import wait_time




def rollout_function(constant_setting, queue_input, queue_result):
    index_chunk, task_space_position = queue_input.get()
    num_chunk, step, dim_ctrl        = task_space_position.shape
    wait_time(const=5, seed=index_chunk)

    env_subclass = constant_setting["env_subclass"]
    config       = constant_setting["config"]
    init_state   = constant_setting["init_state"]

    env = env_subclass(config.env, use_render=False)
    # env.randomize_texture_mode = "static"    # (1) テクスチャをバッチ単位で変更するためper_resetに設定

    # << ------ rollout ------- >>
    robot_state_trajectory  = []
    object_state_trajectory = []
    for n in range(num_chunk):
        robot_position_1seq = []
        object_state_1seq   = []
        env.reset(init_state)
        for t in range(step):
            state = env.get_state()
            robot_position_1seq.append(state.state["robot_position"].value)
            object_state_1seq.append(state.state["object_position"].value)
            # -----
            env.set_ctrl_task_space(task_space_position[n, t])
            env.step()
        state = env.get_state()
        robot_position_1seq.append(state.state["robot_position"].value)
        object_state_1seq.append(state.state["object_position"].value)
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

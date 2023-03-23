import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from forward_model_multiprocessing.ForkedPdb import ForkedPdb
from custom_service import wait_time




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
    object_state_trajectory = []
    for n in range(num_chunk):
        object_state_1seq = []
        env.reset(init_state)
        for t in range(step):
            # img   = env.render()
            state = env.get_state()
            object_state_1seq.append(state.object_position)
            env.set_ctrl_task_space(task_space_position[n, t])
            # env.view()
            env.step()
        object_state_trajectory.append(np.stack(object_state_1seq))
    object_state_trajectory = np.stack(object_state_trajectory)

    # << ---- queue procedure ----- >>
    # ForkedPdb().set_trace()
    queue_result.put((index_chunk, object_state_trajectory)) # 結果とバッチインデックスをキューに入れる
    queue_input.task_done() # キューを終了する

import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
from forward_model_multiprocessing.ForkedPdb import ForkedPdb
from robel_dclaw_env.custom_service import wait_time
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.render.ReturnImage import ReturnImage


def rollout_dataset_collection_fixed_motion(constant_setting, queue_input, queue_result):
    index_chunk, chunked_task_space_position = queue_input.get()
    num_chunk, num_batch, step, _ = chunked_task_space_position.shape
    # print(num_chunk, num_batch, step)
    wait_time(const=5, seed=index_chunk)

    env_subclass = constant_setting["env_subclass"]
    config       = constant_setting["config"]
    init_state   = constant_setting["init_state"]
    dataset_name = constant_setting["dataset_name"]

    env          = env_subclass(config.env, use_render=True)
    repository   = Repository(dataset_dir="./dataset", dataset_name=dataset_name, read_only=False)

    # << ------ rollout ------- >>
    for index_in_chunk in range(num_chunk):
        task_space_position = chunked_task_space_position[index_in_chunk]
        for n in range(num_batch):
            repository.open(filename='domain{}-{}_action{}'.format(index_chunk, index_in_chunk, n))
            image_list = []
            state_list = []
            ctrl_list  = []
            env.reset(init_state)
            for t in range(step):
                img   = env.render()
                state = env.get_state()
                ctrl  = env.set_ctrl_task_space(task_space_position[n, t])

                image_list.append(img)
                state_list.append(state)
                ctrl_list.append(ctrl)

                env.step()

            repository.assign("image", image_list)
            repository.assign("state", state_list)
            repository.assign("ctrl",  ctrl_list)
            repository.close()

    # << ---- queue procedure ----- >>
    # ForkedPdb().set_trace()
    queue_result.put((index_chunk, np.array([[None]]))) # 結果とバッチインデックスをキューに入れる
    queue_input.task_done() # キューを終了する

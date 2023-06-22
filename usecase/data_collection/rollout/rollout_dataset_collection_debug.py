import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
from domain.forward_model_multiprocessing.ForkedPdb import ForkedPdb
from robel_dclaw_env.custom_service import wait_time
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.render.ReturnImage import ReturnImage


def rollout_dataset_collection_debug(constant_setting, task_space_position):


    num_chunk, step, dim_ctrl        = task_space_position.shape
    wait_time(const=5, seed=1)

    env_subclass = constant_setting["env_subclass"]
    config       = constant_setting["config"]
    init_state   = constant_setting["init_state"]
    dataset_name = constant_setting["dataset_name"]
    domain_index = constant_setting["domain_index"]
    ReturnState  = constant_setting["ReturnState"]
    ReturnCtrl   = constant_setting["ReturnCtrl"]

    env          = env_subclass(config.env, use_render=True)
    repository   = Repository(dataset_dir="./dataset", dataset_name=dataset_name, read_only=False)

    # << ------ rollout ------- >>
    object_state_trajectory = []
    for n in range(num_chunk):
        repository.open(filename='domain{}_action{}'.format(domain_index, n))
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
    # queue_result.put((index_chunk, object_state_trajectory)) # 結果とバッチインデックスをキューに入れる
    # queue_input.task_done() # キューを終了する

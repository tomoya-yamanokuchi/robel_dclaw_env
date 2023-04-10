import time
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
from domain.forward_model_multiprocessing.ForkedPdb import ForkedPdb
from custom_service import wait_time, NTD



def rollout_dataset_collection_with_differential_ctrl(constant_setting, queue_input, queue_result):
    # ForkedPdb().set_trace()
    index_ctrl, task_space_differential_ctrl = queue_input.get()
    num_ctrl, step, dim_ctrl = task_space_differential_ctrl.shape
    wait_time(const=5, seed=1)

    env_subclass  = constant_setting["env_subclass"]
    config        = constant_setting["config"]
    init_state    = constant_setting["init_state"]
    TaskSpaceAbs  = constant_setting["TaskSpaceAbs"]
    TaskSpaceDiff = constant_setting["TaskSpaceDiff"]
    dataset_name  = constant_setting["dataset_name"]

    env           = env_subclass(config.env, use_render=True)
    repository    = Repository(dataset_dir="./dataset", dataset_name=dataset_name, read_only=False)

    # << ------ rollout ------- >>
    for n in range(num_ctrl):
        repository.open(filename='domain{}_action{}'.format(index_ctrl, n))
        image_list = []
        state_list = []
        ctrl_list  = []
        env.reset(init_state)
        for t in range(step):
            img                 = env.render()
            state               = env.get_state()
            task_space_position = state.collection["task_space_position"]
            task_space_ctrl     = task_space_position + TaskSpaceAbs(NTD(task_space_differential_ctrl[n, t]))
            ctrl                = env.set_ctrl_task_space(task_space_ctrl)
            ctrl.collection["task_space_diff_position"] = TaskSpaceDiff(NTD(task_space_differential_ctrl[n, t]))

            # ForkedPdb().set_trace()
            image_list.append(img)
            state_list.append(state)
            ctrl_list.append(ctrl)
            env.step()

        # ForkedPdb().set_trace()
        repository.assign(image_list, name="image")
        repository.assign(state_list, name="state")
        repository.assign(ctrl_list, name="ctrl")
        repository.close()


    queue_result.put({
        "index_chunk" : index_ctrl,
    })
    queue_input.task_done()

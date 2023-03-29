import copy
import numpy as np
from forward_model_multiprocessing.ForkedPdb import ForkedPdb



def forward_model_with_queue_example(constant_setting, queue_input, queue_result):
    index_chunk, ctrl = queue_input.get()


    assert len(ctrl.shape) == 3 # (num_sample, horizon, dim)
    assert ctrl.shape[-1]  == 2

    # state = constant_setting["state"]
    # ctrl = copy.deepcopy(ctrl)

    # ForkedPdb().set_trace()
    # cusum_ctrl[:, :, 0] += state[0]
    # cusum_ctrl[:, :, 1] += state[1]

    # path = np.cumsum(ctrl, axis=1)
    simulated_paths = ctrl

    queue_result.put((index_chunk, simulated_paths))
    queue_input.task_done()


import os
import copy
import numpy as np
from forward_model_multiprocessing.ForkedPdb import ForkedPdb

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


def forward_model_progress_check(constant_setting, queue_input, queue_result):
    index_chunk, ctrl = queue_input.get()

    assert len(ctrl.shape) == 3 # (num_sample, horizon, dim)
    assert ctrl.shape[-1]  == 2

    save_fig_dir    = constant_setting["save_fig_dir"]
    iter_outer_loop = constant_setting["iter_outer_loop"]
    iter_inner_loop = constant_setting["iter_inner_loop"]


    simulated_paths = ctrl

    # ForkedPdb().set_trace()

    plt.plot(simulated_paths[0])
    plt.savefig(os.path.join(save_fig_dir, "elite_progress_iterOuter{}_iterInner{}.png".format(iter_outer_loop, iter_inner_loop)))

    queue_result.put((index_chunk, simulated_paths))
    queue_input.task_done()


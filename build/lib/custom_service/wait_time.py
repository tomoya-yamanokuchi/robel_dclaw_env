import time
import numpy as np


def wait_time(seed, const=5, verbose=False):
    np.random.seed(seed)
    wait_time = np.random.rand()*const
    if verbose: print("wait_time = {:.3f}".format(wait_time))
    time.sleep(wait_time)

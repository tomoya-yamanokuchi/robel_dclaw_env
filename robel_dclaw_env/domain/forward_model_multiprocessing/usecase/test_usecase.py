import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from forward_model_multiprocessing.ForwardModelMultiprocessing import ForwardModelMultiprocessing
from forward_model_multiprocessing.utility.rollout_example import rollout_example


if __name__ == '__main__': # <-- must for multiprocessing
    chunked_input = np.random.randn(1, 25, 2)

    multiproc = ForwardModelMultiprocessing(verbose=True)
    for i in range(5):
        print(i)

        result_list = multiproc.run(
            rollout_function = rollout_example,
            constant_setting = None,
            ctrl             = chunked_input,
        )

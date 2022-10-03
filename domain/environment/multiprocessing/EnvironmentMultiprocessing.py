import os
import time
# from line_profiler.line_profiler import LineProfiler
import numpy as np
import multiprocessing
from .EnvironmentConstantSetting import EnvironmentConstantSetting



class EnvironmentMultiprocessing:
    def __init__(self):
        self.method = "fork"
        self.force  = True


    def run(self, function, constant_setting, ctrl):
        assert isinstance(constant_setting,  EnvironmentConstantSetting)
        assert len(ctrl.shape) == 3
        start_time     = time.time()
        num_ctrl       = ctrl.shape[0]
        max_queue_size = min(num_ctrl, int(os.cpu_count()/2))
        multiprocessing.set_start_method(method=self.method, force=self.force)
        queue_result   = multiprocessing.Queue(maxsize=max_queue_size)
        queue_input    = multiprocessing.JoinableQueue(maxsize=max_queue_size)
        for i in range(max_queue_size):
            process = multiprocessing.Process(
                target = function,
                args   = (constant_setting, queue_input, queue_result)
            )
            process.start()
        ctrlList_chunked      = np.array_split(ctrl, indices_or_sections=max_queue_size, axis=0)
        chunked_num_per_queue = [len(ctrlList) for ctrlList in ctrlList_chunked]
        assert sum(chunked_num_per_queue) == num_ctrl

        # print("max_queue_size -> {}".format(max_queue_size))
        # self._print_info(chunked_num_per_queue, ctrl_num)

        for ctrl_index, ctrl_chunked in enumerate(ctrlList_chunked):
            queue_input.put((ctrl_index, ctrl_chunked))
        queue_input.join()

        result_list = self.get_result_list_from_queue(queue_result)
        end_time    = time.time()
        proc_time   = end_time - start_time
        return result_list, proc_time



    def get_result_list_from_queue(self, queue):
        result_tuple = [queue.get() for i in range(queue.qsize())]
        result_dict_list = []
        for (ctrl_index, result) in result_tuple:
            # print("key: {}, val: {}".format(ctrl_index, result))
            result_dict_list.append({
                "ctrl_index" : ctrl_index,
                "result"     : result
            })
        sorted_result_dict_list = sorted(result_dict_list, key=lambda x:x['ctrl_index'])
        return_result = []
        for result_dict in sorted_result_dict_list:
            # print("--- key : {}".format(result_dict["ctrl_index"]))
            for result in result_dict["result"]:
                return_result.append(result)
        return return_result


    def _print_info(self, chunked_num_per_queue, ctrl_num):
        print("--------------------------------------------------------")
        print("  chunked num_per_queue: {}".format(chunked_num_per_queue))
        print("      chunked total_num: {}".format(sum(chunked_num_per_queue)))
        print("     original total_num: {}".format(ctrl_num))
        print("--------------------------------------------------------")

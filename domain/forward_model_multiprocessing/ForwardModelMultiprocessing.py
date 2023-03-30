import os
import time
import numpy as np
import multiprocessing
from .QueueResultAggregation import QueueResultAggregation
from .ProcessTimeInformation import ProcessTimeInformation



class ForwardModelMultiprocessing:
    def __init__(self, verbose=False, result_aggregation=True):
        self.verbose            = verbose
        self.result_aggregation = result_aggregation
        self.queue_agg          = QueueResultAggregation()
        self.proc_time_info     = ProcessTimeInformation(verbose=self.verbose)


    def run(self, rollout_function, constant_setting, ctrl):
        self.proc_time_info.time_start()
        multiprocessing.set_start_method(method="spawn", force=True)
        results   = self._run(rollout_function, constant_setting, ctrl)
        proc_time = self.proc_time_info.time_stop()
        return results, proc_time


    def __get_max_process_num(self, num_chunked_ctrl):
        max_process = min(num_chunked_ctrl, int(os.cpu_count()/2))
        if self.verbose: print("\n [ max_process: {} ] \n".format(max_process))
        return max_process


    def __initialize_queue(self):
        self.queue_input  = multiprocessing.JoinableQueue()
        self.queue_result = multiprocessing.Queue()


    def __start_process(self, max_process, rollout_function, constant_setting):
        self.process = []
        for i in range(max_process):
            process_i = multiprocessing.Process(
                name   = "forward_model_process{}".format(i),
                target = rollout_function,
                args   = (constant_setting, self.queue_input, self.queue_result)
            )
            process_i.start()
            self.process.append(process_i)


    def __stop_process(self):
        for process_i in self.process:
            process_i.join()
            process_i.close()


    def __put_control_into_queue(self, max_process, ctrl):
        chunked_ctrl = np.array_split(ctrl, max_process, axis=0)
        for index_chunk, ctrl_i in enumerate(chunked_ctrl):
            self.queue_input.put((index_chunk, ctrl_i))
        if self.verbose: print(" self.queue_input.qsize()",  self.queue_input.qsize())


    def _run(self, rollout_function, constant_setting, ctrl):
        self.__initialize_queue()
        max_process = self.__get_max_process_num(len(ctrl))
        self.__start_process(max_process, rollout_function, constant_setting)
        self.__put_control_into_queue(max_process, ctrl)
        self.queue_input.join()
        # if self.result_aggregation: result = self.queue_agg.aggregate(self.queue_result)
        result = self.queue_agg.aggregate(self.queue_result)
        self.queue_input.close()
        self.queue_result.close()
        self.__stop_process()
        return result

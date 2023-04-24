from pprint import pprint
import pandas as pd
import numpy as np



class QueueResultAggregation:
    def _initialize_aggregation_result(self, forward_results):
        keys = list(forward_results[0].keys())
        aggregation_results = {}
        for key in keys:
            aggregation_results[key] = []
        return aggregation_results


    def _append_reuslt(self, aggregation_results, result_tuple):
        keys = list(aggregation_results.keys())
        for result in result_tuple:
            for key in keys:
                aggregation_results[key].append(result[key])
        return aggregation_results


    def aggregate(self, queue):
        result_tuple = [queue.get() for i in range(queue.qsize())]
        aggregation_results = self._initialize_aggregation_result(result_tuple)
        aggregation_results = self._append_reuslt(aggregation_results, result_tuple)
        aggregation_results = self._concat_result(aggregation_results)
        return aggregation_results


    def _concat_result(self, aggregation_results):
        sort_func = {
            "index_chunk"             : self.__sort_index_chunk,
            "robot_state_trajectory"  : self.__sort_robot_state_trajectory,
            "object_state_trajectory" : self.__sort_object_state_trajectory,
            "task_space_ctrl"         : self.__sort_task_space_ctrl,
            "ctrl_t"                  : self.__sort_ctrl_t,
            "state"                   : self.__sort_state,
        }
        index_chunk_sorted = np.argsort(aggregation_results['index_chunk'])
        for key in list(aggregation_results.keys()):
            aggregation_results[key] = sort_func[key](aggregation_results[key], index_chunk_sorted)
        return aggregation_results


    def __sort_index_chunk(self, aggregation_results, index_chunk_sorted):
        x = list(np.take(aggregation_results, index_chunk_sorted))
        return x


    def __sort_robot_state_trajectory(self, aggregation_results, index_chunk_sorted):
        '''
        numpy の shape 関連の warningの出処
        '''
        x = np.take(aggregation_results, index_chunk_sorted, axis=0)
        # import ipdb; ipdb.set_trace()
        # assert len(x.shape) == 4
        # assert x.shape[1] == 1
        # x = np.squeeze(x, axis=1)
        # import ipdb; ipdb.set_trace()
        return np.concatenate(x)


    def __sort_object_state_trajectory(self, aggregation_results, index_chunk_sorted):
        x = np.take(aggregation_results, index_chunk_sorted, axis=0)
        # assert len(x.shape) == 4
        # assert x.shape[1] == 1
        # x = np.squeeze(x, axis=1)
        return np.concatenate(x)


    def __sort_state(self, aggregation_results, index_chunk_sorted):
        # import ipdb; ipdb.set_trace()
        assert len(aggregation_results) == 1
        return aggregation_results[0]

    def __sort_ctrl_t(self, aggregation_results, index_chunk_sorted):
        # import ipdb; ipdb.set_trace()
        assert len(aggregation_results) == 1
        return aggregation_results[0]


    def __sort_task_space_ctrl(self, aggregation_results, index_chunk_sorted):
        # import ipdb; ipdb.set_trace()
        assert len(aggregation_results) == 1
        return aggregation_results[0]

from pprint import pprint
import pandas as pd
import numpy as np

class QueueResultAggregation:
    '''
        queueの結果をインデックスの順番に従って並び替えて取り出す
    '''

    def _extract_result_as_list_of_dict(self, queue):
        result_tuple     = [queue.get() for i in range(queue.qsize())]
        result_dict_list = []
        for (index_chunk, result) in result_tuple:
            result_dict_list.append({
                "index_chunk" : index_chunk,
                "result"     : result
            })
        df_result = pd.DataFrame(result_dict_list)
        return df_result


    def aggregate(self, queue):
        df_result = self._extract_result_as_list_of_dict(queue)
        df_result = df_result.sort_values(by="index_chunk")
        df_result.reset_index(drop=True, inplace=True)
        return np.concatenate(df_result["result"])


    def get_single_result(self, queue):
        index_chunk, result = queue.get()
        return result

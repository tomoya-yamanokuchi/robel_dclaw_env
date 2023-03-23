

import numpy as np


def pseudo_rollout(constant_setting, queue_input, queue_result):
    index_chunk, task_space_position = queue_input.get()
    queue_result.put((index_chunk, np.array([[None]]))) # 結果とバッチインデックスをキューに入れる
    queue_input.task_done() # キューを終了する

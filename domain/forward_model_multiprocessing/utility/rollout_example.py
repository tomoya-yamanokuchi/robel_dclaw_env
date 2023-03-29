import time
import numpy as np
from ..ForkedPdb import ForkedPdb



def rollout_example(constant_setting, queue_input, queue_result):
    index_chunk, chunked_ctrl_input = queue_input.get()

    y = chunked_ctrl_input**2

    time.sleep(0.01)


    queue_result.put((index_chunk, y)) # 結果とバッチインデックスをキューに入れる
    queue_input.task_done() # キューを終了する
    # queue_input

from dataclasses import dataclass
from mimetypes import init
import numpy as np

'''
・Dclaw環境に状態を与える時に使用するクラスです
・与えるべき状態のルールが記述されています
'''

@dataclass(frozen=True)
class ValveFeedState:
    '''
    modeについて：
        永続化するときには系列になった値オブジェクトとして保存したいが，系列とステップごととで
        shapeに対するassetの掛け方が変化するようしたい．このassertの掛け方を判断するのがmode．
        - mode = "step": ステップデータとしてのshpaeをassert
        - mode = "sequence": 系列データとしてのshapeをassert
    '''
    task_space_positioin : np.ndarray
    object_position      : np.ndarray
    robot_velocity       : np.ndarray
    object_velocity      : np.ndarray


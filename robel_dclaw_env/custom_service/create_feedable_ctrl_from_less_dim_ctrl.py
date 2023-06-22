import numpy as np



'''
目的: 全体のうち一部の次元だけをサンプルしたactionで制御したい
→ 不完全なctrlから完全次元なctrlを作りたい
- 制御入力は何次元か？
- サンプルされた制御入力は何か？
- 興味のある次元はどれか？
'''


def create_feedable_ctrl_from_less_dim_ctrl(
        dim_totoal_ctrl : int,
        task_space_differential_ctrl : np.ndarray,
        dimension_of_interst: list,
    ):
    assert np.array(dimension_of_interst).max() <= dim_totoal_ctrl
    num_sample, step, _ = task_space_differential_ctrl.shape
    ctrl = np.zeros([num_sample, step, dim_totoal_ctrl])
    ctrl[:, :, dimension_of_interst] += task_space_differential_ctrl
    return ctrl

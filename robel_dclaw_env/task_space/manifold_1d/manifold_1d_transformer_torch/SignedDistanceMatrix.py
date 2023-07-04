import copy
import pathlib
import torch
from .ReferencePosition import ReferencePosition

# 上位ディレクトリからのインポート
import sys, pprint
p_file = pathlib.Path(__file__)
path_environment = "/".join(str(p_file).split("/")[:-2])
sys.path.append(path_environment)
from robel_dclaw_env.domain.environment.kinematics import ForwardKinematics
from .replace_zero_point import replace_zero_point_with_given_value, replace_zero_point_with_one
from .save_matrix_as_heatmap import save_matrix_as_heatmap
from robel_dclaw_env.custom_service import to_numpy


class SignedDistanceMatrix:
    def __init__(self, is_plot:bool=True):
        self.dim_task_space = 3
        self.is_plot        = is_plot


    def create(self, task_space_position, reference_task_space_position):
        assert len(task_space_position.shape)           == 1 # (num_data,)
        assert len(reference_task_space_position.shape) == 1 # (num_reference,)
        signed_distance_matrix_with_zero = self.__signed_distance_matrix(task_space_position, reference_task_space_position)
        signed_distance_matrix           = self.__replace_zero_point(signed_distance_matrix_with_zero)
        assert (signed_distance_matrix == 0).sum() == 0 # ゼロ要素が無いことを確認
        if self.is_plot: save_matrix_as_heatmap(x=to_numpy(signed_distance_matrix), save_path="./signed_distance_matrix_torch.png")
        return signed_distance_matrix


    def __signed_distance_matrix(self, task_space_position, reference_task_space_position):
        distance_matrix = task_space_position.reshape(-1, 1) - reference_task_space_position.reshape(1, -1) # referenceとの差を計算
        return torch.sign(distance_matrix) # referenceとの差の符号を取得


    def __replace_zero_point(self, x):
        '''
            0 があると後の計算（どこで符号が切り替わるかの判定）で困るので1か-1に変換しておく
        '''
        x = replace_zero_point_with_given_value(x, index= 0, value= 1) # 初期reference点と同一で0になっている部分を1に置き換え
        x = replace_zero_point_with_given_value(x, index=-1, value=-1) # 最終reference点と同一で0になっている部分を-1に置き換え
        x = replace_zero_point_with_one(x)                             # それ以外の0に一致する点は全て1に置き換え
        return x



if __name__ == '__main__':
    import copy
    import pathlib
    import numpy as np
    import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd()))
    sys.path.insert(0, './robel_dclaw_env')
    from task_space.manifold_1d.ReferencePosition import ReferencePosition
    from task_space.manifold_1d.create_cyclic_data import create_cyclic_data
    from robel_dclaw_env.domain.environment.kinematics.ForwardKinematics import ForwardKinematics
    from task_space.AbstractTaskSpaceTransformer import AbstractTaskSpace
    from ReferenceTaskSpacefromEndEffector import ReferenceTaskSpacefromEndEffector

    min                             = 0.0
    max                             = 1.0
    forward_kinematics              = ForwardKinematics()
    reference_joint_position        = ReferencePosition().augument_reference_joint_position()
    reference_end_effector_position = forward_kinematics.calc_1claw(reference_joint_position)
    reference_end_effector_position = create_cyclic_data(reference_end_effector_position) # 中間点を補完する際にはtask_spaceとして閉じている必要がある
    # reference_task_space_position   = _create_reference_task_space_position()
    # num_claw                        = 3

    reference_task_space_position = ReferenceTaskSpacefromEndEffector().create(reference_end_effector_position)
    task_space_position           = torch.random.randn(5).clip(0, 1)
    signed_distance_matrix        = SignedDistanceMatrix().create(task_space_position, reference_task_space_position)
    print(signed_distance_matrix)

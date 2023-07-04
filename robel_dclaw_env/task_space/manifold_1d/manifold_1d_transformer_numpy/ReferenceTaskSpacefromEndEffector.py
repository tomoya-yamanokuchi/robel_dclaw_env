import copy
import torch
import numpy as np

class ReferenceTaskSpacefromEndEffector:
    def create(self, reference_end_effector_position):
        cumsum_euclidean_distance            = self.__cumulative_euclidean_distance(reference_end_effector_position)
        normalized_cumsum_euclidean_distance = self.__normalize(cumsum_euclidean_distance)
        reference_task_space_position        = self.__add_initial_point(normalized_cumsum_euclidean_distance)
        return reference_task_space_position


    def __cumulative_euclidean_distance(self, reference_end_effector_position):
        num_data, dim_xyz         = reference_end_effector_position.shape; assert dim_xyz == 3
        diff_for_each_data        = np.diff(reference_end_effector_position, n=1, axis=0) # 各データ点ごとの差分を各次元で計算
        euclidean_distance        = np.sqrt(np.sum(diff_for_each_data**2, axis=-1))       # [差分を2乗] + [次元を総和] + [平方根] を取ることで各データ点間のユークリッド距離を計算
        cumsum_euclidean_distance = np.cumsum(euclidean_distance, axis=0)                 # 1つめ目のデータ点を起点とした各データ点間までのユークリッド距離の累積和を計算
        return cumsum_euclidean_distance


    def __normalize(self, cumsum_euclidean_distance):
        self.max_euclidean_distance          = cumsum_euclidean_distance.max()                    # ユークリッド距離の累積和の最大値を取得
        normalized_cumsum_euclidean_distance = cumsum_euclidean_distance / self.max_euclidean_distance # ユークリッド距離の累積和を最大値で割ることで[0, 1]の範囲に正規化
        return normalized_cumsum_euclidean_distance


    def __add_initial_point(self, normalized_cumsum_euclidean_distance):
        cyclic_normalized_cumsum_euclidean_distance = np.hstack((np.zeros(1), normalized_cumsum_euclidean_distance)) # 最初の点までをつなぐ開始点として0を追加
        return cyclic_normalized_cumsum_euclidean_distance


    def get_max_euclidean_distance(self):
        return copy.deepcopy(self.max_euclidean_distance)



if __name__ == '__main__':
    import copy
    import pathlib
    import numpy as np
    import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd()))
    sys.path.insert(0, './robel_dclaw_env')
    from task_space.manifold_1d.ReferencePosition import ReferencePosition
    from task_space.manifold_1d.create_cyclic_data import create_cyclic_data
    from robel_dclaw_env.domain.environment.kinematics import ForwardKinematics, to_tensor
    from task_space.AbstractTaskSpaceTransformer import AbstractTaskSpace

    min                             = 0.0
    max                             = 1.0
    forward_kinematics              = ForwardKinematics()
    reference_joint_position        = ReferencePosition().augument_reference_joint_position()
    reference_end_effector_position = forward_kinematics.calc_1claw(to_tensor(reference_joint_position))
    reference_end_effector_position = create_cyclic_data(reference_end_effector_position) # 中間点を補完する際にはtask_spaceとして閉じている必要がある
    # reference_task_space_position   = _create_reference_task_space_position()
    # num_claw                        = 3

    create_ref_task = ReferenceTaskSpacefromEndEffector()
    task_space = create_ref_task.create(reference_end_effector_position)
    import ipdb; ipdb.set_trace()

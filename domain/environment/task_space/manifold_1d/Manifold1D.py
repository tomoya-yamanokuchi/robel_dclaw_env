import copy
import pathlib
import numpy as np
from .ReferencePosition import ReferencePosition
from custom_service import data_shape_formating

# 上位ディレクトリからのインポート
# import sys, pprint
# p_file = pathlib.Path(__file__)
# path_environment = "/".join(str(p_file).split("/")[:-2])
# sys.path.append(path_environment)

from .TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold as TaskSpaceValueObject
from .EndEffectorPositionValueObject import EndEffectorPositionValueObject as EndEffectorValueObject
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.environment.kinematics.ForwardKinematics import ForwardKinematics
from  domain.environment.task_space.AbstractTaskSpace import AbstractTaskSpace
from custom_service import NTD


'''
[方策のobservationとしてtask_space_positionを使う場合の問題]
- エンドエフェクタの３次元空間中において行動空間を１次元空間のようにして拘束する場合
　定義した task_space 上で0と1のような境界部分で非連続性があるためモデル/方策学習に望ましくない
- ロボットの状態が task_space 上に必ずしも位置しているとは限らないためobservationに対して近似が必要になる
　（バルブと接触状態にある場合には task_space 上から指が外れてしまうことは頻発する）

[行動決定のための擬似的なobservationとしてtask_space_positionを使用する理由]
- task_space_positionを絶対値の空間で自由に決定してしまうと1次元の拘束が意味を持たなくなる
- 例えば[0, 1]の範囲のtask_spaceなのに0.1から次に急に0.4とかの離れた行動をとってしまうと
　明らかに1次元拘束の軌道上から外れてショートカットしたような軌道になってしまう
- これを避けるためには差分入力の形で1stepに動けるtask_space上の距離を制限して行動を実行する必要がある
- 差分行動を決定するためには現在のtask_space_positionが必要となる
- しかし，エンドエフェクタ位置から対応するtask_space_positionを得る逆写像は容易ではない
　（エンドエフェクタ位置が厳密にtask_spaceの軌道上に乗っかっている保証はないため）
- そこであらかじめtask_space上でobservationとなる点の候補を多めに生成しておき，それらとのノルムを測り
　最近棒のものを擬似的な観測として差分制御入力を決定するために用いる
'''


class Manifold1D(AbstractTaskSpace):
    def __init__(self):
        self.min                             = 0.0
        self.max                             = 1.0
        self.forward_kinematics              = ForwardKinematics()
        self.reference_joint_position        = ReferencePosition().augument_reference_joint_position()
        self.reference_end_effector_position = self.forward_kinematics.calc_1claw(self.reference_joint_position)
        self.reference_end_effector_position = self._create_cyclic_data(self.reference_end_effector_position) # 中間点を補完する際にはtask_spaceとして閉じている必要がある
        self.reference_task_space_position   = self._create_reference_task_space_position()
        self.num_claw                        = 3


    def _create_cyclic_data(self, x):
        return np.concatenate([x, x[:1]], axis=0)


    def _create_reference_task_space_position(self):
        diff_reference_end_effector                     = np.diff(self.reference_end_effector_position, n=1, axis=0)          # xyzの各次元での差分
        euclidean_distance_for_each_reference           = np.sqrt(np.sum(diff_reference_end_effector**2, axis=-1))            # 差分を2乗してxyzの次元を総和して平方根を取ることで各点間のユークリッド距離を計算
        cumulative_euclidean_distance                   = np.cumsum(euclidean_distance_for_each_reference, axis=0)            # ユークリッド距離の累積和を計算
        self.max_euclidean_distance                     = cumulative_euclidean_distance.max()                                 # ユークリッド距離の累積和の最大値を取得
        normalized_cumulative_euclidean_distance        = cumulative_euclidean_distance / self.max_euclidean_distance         # ユークリッド距離の累積和を最大値で割ることで[0, 1]の範囲に正規化
        cyclic_normalized_cumulative_euclidean_distance = np.hstack((np.zeros(1), normalized_cumulative_euclidean_distance))  # 最初の点までをつなぐ開始点として0を追加
        return cyclic_normalized_cumulative_euclidean_distance


    # @abstractmethod
    def task2end(self, task_space_position: TaskSpaceValueObject):
        end_effector_position = [self._task2end_1claw(x) for x in np.split(task_space_position.value, self.num_claw, axis=-1)]
        return EndEffectorValueObject(NTD(np.concatenate(end_effector_position, axis=-1)))


    def _task2end_1claw(self, task_space_position):
        '''
        - 中間点を補完する際に，単純にreference_task_space_positionとの差だけで補完に使用する2点を決定してしまうと
        必ず補完の空洞になる領域が発生してしまい，連続的な行動空間を生成できない（点同士が距離が離れている領域付近で欠落する）
        - なのでnp.argsort(dist_matrix)から上位2つの近傍点を取得するだけでは正しく補完できない
        - 理由：reference_task_space_positionにある各点同士の距離は一定ではないため
        '''
        assert len(task_space_position.shape) == 3
        assert task_space_position.shape == (1,1,1)
        distance_matrix                    = task_space_position.reshape(-1, 1) - self.reference_task_space_position.reshape(1, -1)              # referenceとの差を計算
        signed_distance_matrix             = np.sign(distance_matrix)                                                                            # referenceとの差をの符号を取得
        signed_distance_matrix             = self._convet_zero_to_plus1_or_minus1_for_calculating_difference_of_sign(signed_distance_matrix)     # 0があると後の計算で困るので1か-1に変換しておく
        index_sign_changeed                = np.nonzero(np.diff(signed_distance_matrix, n=1, axis=-1))[1]                                        # referenceとの差で符号関係が変化する点を探す（以下，以上の関係が変化する点）
        index_top2_nearest_neighbor        = np.concatenate((index_sign_changeed.reshape(-1, 1), index_sign_changeed.reshape(-1, 1)+1), axis=-1) # 符号関係が変化する位置から補完に用いる2点を取得
        top2_nearest_task_space_position   = np.take(self.reference_task_space_position, index_top2_nearest_neighbor)                            # 補完に用いるtask_space_space_positionを取得
        top2_nearest_end_effector_position = np.take(self.reference_end_effector_position, index_top2_nearest_neighbor, axis=0)                  # 補完に用いるend_effector_positionを取得
        direction_vector                   = np.diff(top2_nearest_end_effector_position, n=1, axis=1)                                            # 補完に用いる方向ベクトルを計算
        direction_vector_squeezed          = np.squeeze(direction_vector, axis=1)                                                                # データ形状を整形
        unit_direction_vector              = direction_vector_squeezed / np.linalg.norm(direction_vector_squeezed, axis=-1, keepdims=True)       # 方向ベクトルと同一方向の単位ベクトルを計算
        t                                  = np.abs(task_space_position - top2_nearest_task_space_position[:,0]) * self.max_euclidean_distance   # task_space_positionを表すための媒介変を計算
        end_effector_position              = top2_nearest_end_effector_position[:, 0] + t.reshape(-1, 1) * unit_direction_vector                 # 補完点を計算
        return end_effector_position


    def _convet_zero_to_plus1_or_minus1_for_calculating_difference_of_sign(self, input_array):
        # edge index preprocesssing
        input_array_inserted_minus1_at_index0 = self._insert_specific_value_instead_of_zero_dependig_on_the_index(input_array, value=1, index=0)
        input_array_inserted_minus1_and_plus1 = self._insert_specific_value_instead_of_zero_dependig_on_the_index(input_array_inserted_minus1_at_index0, value=-1, index=-1)
        # non-edge index preprocesssing
        index_zero_elements = np.where(input_array_inserted_minus1_and_plus1==0)
        input_array_inserted_minus1_and_plus1[index_zero_elements] = 1
        return input_array_inserted_minus1_and_plus1


    def _insert_specific_value_instead_of_zero_dependig_on_the_index(self, input_array, value, index):
        input_array = copy.deepcopy(input_array)
        assert (len(input_array.shape) == 2) and (input_array.shape[1] == self.reference_task_space_position.shape[0])
        zero_element_index = np.where(input_array[:, index]==0)[0]
        input_array[zero_element_index, index] = value
        return input_array


    # @abstractmethod
    def end2task(self, end_effector_position: EndEffectorValueObject):
        task_space_position = [self._end2task_1claw(x) for x in np.split(end_effector_position.value, self.num_claw, axis=-1)]
        return TaskSpaceValueObject(NTD(np.concatenate(task_space_position, axis=-1)))


    def _end2task_1claw(self, end_effector_position):
        assert end_effector_position.shape == (1, 1, 3) # １つのendeffector座標を比較するため
        end_effector_position  = end_effector_position.squeeze(0)
        distance               = np.linalg.norm(self.reference_end_effector_position - end_effector_position, axis=-1)
        index_minimum_distance = np.argsort(distance)[0]
        nearest_reference      = self.reference_task_space_position[index_minimum_distance]
        return np.array([nearest_reference])




if __name__ == '__main__':

    from custom_service import visualization as vis
    taskspace = Manifold1D()

    t = np.linspace(start=0.0, stop=2.0, num=200)
    t = t.reshape(-1, 1)
    t = np.tile(t, (1, 3))

    end = taskspace.calc(t)

    # vis.scatter_3d(end[:, :3])
    vis.scatter_3d_animation(end[:, :3], num_history=100, interval=10)

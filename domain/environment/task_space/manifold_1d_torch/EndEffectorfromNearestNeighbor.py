import torch


class EndEffectorfromNearestNeighbor:
    def __init__(self, reference_end_effector_position, reference_task_space_position, index_top2_nearest_neighbor, max_euclidean_distance):
        self.reference_end_effector_position = reference_end_effector_position
        self.reference_task_space_position   = reference_task_space_position
        self.index_top2_nearest_neighbor     = index_top2_nearest_neighbor
        self.max_euclidean_distance          = max_euclidean_distance


    def get(self, task_space_position):
        end_effector_position_top1         = torch.index_select(self.reference_end_effector_position, dim=0, index=self.index_top2_nearest_neighbor[:, 0])
        end_effector_position_top2         = torch.index_select(self.reference_end_effector_position, dim=0, index=self.index_top2_nearest_neighbor[:, 1])
        top2_nearest_end_effector_position = torch.cat((end_effector_position_top1.unsqueeze(1), end_effector_position_top2.unsqueeze(1)), dim=1)
        unit_direction_vector              = self.__unit_direction_vecotor(top2_nearest_end_effector_position)
        # ---------------------------------
        nearest_neighbor_task_space_position               = torch.take(self.reference_task_space_position,   self.index_top2_nearest_neighbor[:, 0])
        relative_task_space_distance_with_nearest_neighbor = torch.abs(task_space_position - nearest_neighbor_task_space_position)
        nearest_neighbor_end_effector_position             = top2_nearest_end_effector_position[:, 0]
        end_effector_posiiton                              = nearest_neighbor_end_effector_position + \
            ((relative_task_space_distance_with_nearest_neighbor.reshape(-1, 1) * unit_direction_vector) * self.max_euclidean_distance)
        return end_effector_posiiton


    def __unit_direction_vecotor(self, top2_nearest_end_effector_position):
        '''
        input :
            top2_nearest_end_effector_position: shape = (num_query, 2, 3) = (num_query, num_top2, dim_end_effector)
        '''
        direction_vector      = torch.diff(top2_nearest_end_effector_position, n=1, axis=1) # 補完に用いる方向ベクトルを計算: (num_query, 1, 3)
        direction_vector      = torch.squeeze(direction_vector, axis=1)                     # torch.diff では diff した次元が残るのでなくす: (num_query, 3)
        norm_direction_vector = torch.linalg.norm(direction_vector, axis=-1, keepdims=True) # ベクトルのノルムを計算: (num_query, 1)
        unit_direction_vector = direction_vector / norm_direction_vector                    # 方向ベクトルと同一方向の単位ベクトルをxyzそれぞれで計算: (num_query, 3)
        return unit_direction_vector


    def _debug_get_unit_direction_vector(self):
        end_effector_position_top1         = torch.index_select(self.reference_end_effector_position, dim=0, index=self.index_top2_nearest_neighbor[:, 0])
        end_effector_position_top2         = torch.index_select(self.reference_end_effector_position, dim=0, index=self.index_top2_nearest_neighbor[:, 1])
        top2_nearest_end_effector_position = torch.cat((end_effector_position_top1.unsqueeze(1), end_effector_position_top2.unsqueeze(1)), dim=1)
        unit_direction_vector              = self.__unit_direction_vecotor(top2_nearest_end_effector_position)
        return unit_direction_vector

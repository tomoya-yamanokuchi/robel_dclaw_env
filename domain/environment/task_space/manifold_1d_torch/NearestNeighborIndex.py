import torch
from .save_matrix_as_heatmap import save_matrix_as_heatmap
from torch_numpy_converter import to_numpy


class NearestNeighborIndex:
    def __init__(self, is_plot=True, verbose=False):
        self.is_plot = is_plot
        self.verbose = verbose


    def get(self, signed_distance_matrix):
        index_sign_change           = self.__sign_change_point(signed_distance_matrix)
        index_top2_nearest_neighbor = self.__top2_nearest_neighbor_index(index_sign_change)
        return index_top2_nearest_neighbor
        # return index_sign_change


    def __sign_change_point(self, signed_distance_matrix):
        # referenceとの差で符号関係が変化する点を探す（以下，以上の関係が変化する点）
        num_query, num_reference    = signed_distance_matrix.shape
        if self.verbose: print("[num_query, num_reference] = [{}, {}]".format(num_query, num_reference))
        diff_signed_distance_matrix = torch.diff(signed_distance_matrix, n=1, axis=-1)
        if self.is_plot: save_matrix_as_heatmap(x=to_numpy(diff_signed_distance_matrix), save_path="./diff_signed_distance_matrix.png")
        index_sign_change = torch.argmin(diff_signed_distance_matrix, dim=1)
        return index_sign_change


    def __top2_nearest_neighbor_index(self, index_sign_change):
        # 符号関係が変化する位置から補完に用いる2点のインデックスを作成
        return torch.cat((index_sign_change.reshape(-1, 1), index_sign_change.reshape(-1, 1)+1), dim=-1)

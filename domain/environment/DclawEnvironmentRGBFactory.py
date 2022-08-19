import copy
import numpy as np
from typing import List


'''
・ロボットやオブジェクト・背景などの環境内の色を定義するクラスです
・configファイル内のenv_colorパラメータによって指定します
'''


class DclawEnvironmentRGBFactory:
    def __init__(self, geom_names: List[int]):
        self._set_semactic_grouped_geom(geom_names)
        self._set_rgb()

    def _set_semactic_grouped_geom(self, geom_names: List[int]):
        self.geom_names = geom_names
        self.geom_names_dict = {
            "base_plate": [s for s in geom_names if 'base_plate' in s ],
            "floor"     : [s for s in geom_names if 'floor' in s ],
            "tip"       : [s for s in geom_names if 'finger' in s],
            # "valve"     : [s for s in geom_names if 'valve' in s],
            "valve_3fin_handle_1" : [s for s in geom_names if 'valve_3fin_handle_1' in s],
            "valve_3fin_handle_2" : [s for s in geom_names if 'valve_3fin_handle_2' in s],
            "valve_3fin_handle_3" : [s for s in geom_names if 'valve_3fin_handle_3' in s],
            "valve_3fin_center"   : [s for s in geom_names if 'valve_3fin_center' in s],
            "robot"     : [s for s in geom_names if \
                            ('floor' not in s) and \
                            ('finger' not in s) and \
                            ('valve' not in s) and \
                            ('valve' not in s) and \
                            ('base_plate' not in s) \
                          ]
        }


    def create(self, env_color: str):
        return self.rgb[env_color]


    def _create_rgb_dict(self, rgb_dict):
        rgb = dict()
        for key, val in rgb_dict.items():
            names = self.geom_names_dict[key]
            for geom_name  in names:
                print(key, geom_name, val)
                rgb[geom_name] = np.array(val)
        return rgb


    def _set_rgb(self):
        self.rgb = dict()

        self.rgb["rgb_valve"] = self._create_rgb_dict({
            "base_plate"            : [200, 200, 200],
            "floor"                 : [200, 200, 200],
            "valve_3fin_handle_1"   : [255, 0, 0],
            "valve_3fin_handle_2"   : [0, 255, 0],
            "valve_3fin_handle_3"   : [0, 0, 255],
            "valve_3fin_center"     : [255, 255, 255],
            "robot"                 : [ 38,  38,  38],
            "tip"                   : [255, 127,   0]
        })

        self.rgb["overview"] = copy.deepcopy(self.rgb["rgb_valve"])
        self.rgb["random"]   = copy.deepcopy(self.rgb["rgb_valve"])

        # ----- 追加で環境の色セットを追加できます -----
        #  以下のフォーマットです
        # self.rgb[env_color] = self._create_rgb_dict({
        #     "base_plate"            : - ,
        #     "floor"                 : - ,
        #     "valve_3fin_handle_1"   : - ,
        #     "valve_3fin_handle_2"   : - ,
        #     "valve_3fin_handle_3"   : - ,
        #     "valve_3fin_center"     : - ,
        #     "robot"                 : - ,
        #     "tip"                   : - ,
        # })
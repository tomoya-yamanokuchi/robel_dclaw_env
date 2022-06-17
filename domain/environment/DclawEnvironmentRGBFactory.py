import numpy as np
from typing import List


'''
・ロボットやオブジェクト・背景などの環境内の色を定義するクラスです
・configファイル内のenv_colorパラメータによって指定します
'''


class DclawEnvironmentRGBFactory:
    def create(self, env_color: str, geom_names: List[int]):
        self.geom_names = geom_names
        self.geom_names_dict = {
            "floor"  : [s for s in geom_names if 'floor' in s ],
            "tip"    : [s for s in geom_names if 'finger' in s],
            "valve"  : [s for s in geom_names if 'valve' in s],
            "robot"  : [s for s in geom_names if \
                            ('floor' not in s) and ('finger' not in s) and ('valve' not in s) and ('valve' not in s)]
        }
        return self._set_geom_rgb(env_color)


    def create_rgb_dict(self, rgb_dict):
        rgb = dict()
        for key, val in rgb_dict.items():
            names = self.geom_names_dict[key]
            for geom_name  in names:
                print(key, geom_name, val)
                rgb[geom_name] = np.array(val)
        return rgb


    def _set_geom_rgb(self, env_color):
        assert env_color is not None

        rgb = dict()

        if env_color == "white":
            rgb = self.create_rgb_dict({
                "floor" : [ 27, 176,  27],
                "valve" : [255, 255, 255],
                "robot" : [ 38,  38,  38],
                "tip"   : [255, 127,   0]
            })

        elif env_color == "transparent":
            rgb = self.create_rgb_dict({
                "floor" : [ 27, 176,  27],
                "valve" : [255, 255, 255],
                "robot" : [ 38,  38,  38],
                "tip"   : [255, 127,   0]
            })


        elif env_color == "pseudo_black":
            rgb = self.create_rgb_dict({
                "floor" : [20,  20,  20],
                "valve" : [20,  20,  20],
                "robot" : [38,  38,  38],
                "tip"   : [255, 127,   0]
            })

        elif env_color == "pseudo_red":
            rgb = self.create_rgb_dict({
                "floor" : [213, 0, 0],
                "valve" : [213, 0, 0],
                "robot" : [38,  38,  38],
                "tip"   : [255, 127,   0]
            })

        elif env_color == "pseudo_green":
            rgb = self.create_rgb_dict({
                "floor" : [27, 176,  27],
                "valve" : [27, 176,  27],
                "robot" : [38,  38,  38],
                "tip"   : [255, 127,   0]
            })

        elif env_color == "pseudo_blue":
            rgb = self.create_rgb_dict({
                "floor" : [0, 145, 234],
                "valve" : [0, 145, 234],
                "robot" : [38,  38,  38],
                "tip"   : [255, 127,   0]
            })


        elif env_color == "red":
            rgb = self.create_rgb_dict({
                "floor" : [27, 176,  27],
                "valve" : [255, 74, 64],
                "robot" : [38,  38,  38],
                "tip"   : [255,127,   0]
            })

        elif env_color == "green":
            rgb = self.create_rgb_dict({
                "floor" : [27, 176,  27],
                "valve" : [177, 225, 113],
                "robot" : [38,  38,  38],
                "tip"   : [255,127,   0]
            })

        elif env_color == "blue":
            rgb = self.create_rgb_dict({
                "floor" : [27, 176,  27],
                "valve" : [ 0,  42, 174],
                "robot" : [38,  38,  38],
                "tip"   : [255,127,   0]
            })

        elif env_color == "random":
            '''
                どんな色にしようがランダム化されて意味ないが，一応定義しておかないと
                この段階でエラーが出るのでrandomも定義しておく
            '''
            rgb = self.create_rgb_dict({
                "floor" : [ 27, 176,  27],
                "valve" : [255, 255, 255],
                "robot" : [ 38,  38,  38],
                "tip"   : [255, 127,   0]
            })

        else:
            raise NotImplementedError()

        return rgb

import numpy as np
from mujoco_py.modder import TextureModder


'''
・mujocoのTextureModderクラスを継承した拡張クラスです
・テクスチャのランダム化の再現性の確保・ステップごとのランダム化の有無を実現するためのクラスです
'''


class myTextureModder(TextureModder):
    def __init__(self, sim, random_state=None):
         super().__init__(sim, random_state)


    def my_rand_all(self, name):
        checker_rgb1, checker_rgb2                         = self.my_rand_checker(name)
        gradient_rgb1, gradient_rgb2, gradient_vertical    = self.my_rand_gradient(name)
        rgb_rgb                                            = self.my_rand_rgb(name)
        noise_rgb1, noise_rgb2, noise_mask, noise_fraction = self.my_rand_noise(name)

        texture_info = {
            # ----------------------
            "checker_rgb1"      : checker_rgb1,
            "checker_rgb2"      : checker_rgb2,
            # ----------------------
            "gradient_rgb1"     : gradient_rgb1,
            "gradient_rgb2"     : gradient_rgb2,
            "gradient_vertical" : gradient_vertical,
            # ----------------------
            "rgb_rgb"           : rgb_rgb,
            # ----------------------
            "noise_rgb1"        : noise_rgb1,
            "noise_rgb2"        : noise_rgb2,
            "noise_mask"        : noise_mask,
            "noise_fraction"    : noise_fraction,
        }
        return texture_info

    def my_rand_checker(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        self.set_checker(name, rgb1, rgb2)
        return rgb1, rgb2


    def my_rand_gradient(self, name):
        rgb1, rgb2 = self.get_rand_rgb(2)
        vertical = bool(self.random_state.uniform() > 0.5)
        self.set_gradient(name, rgb1, rgb2, vertical=vertical)
        return rgb1, rgb2, vertical

    def my_rand_rgb(self, name):
        rgb = self.get_rand_rgb()
        self.set_rgb(name, rgb)
        return rgb


    def my_rand_noise(self, name):
        fraction = 0.1 + self.random_state.uniform() * 0.8
        rgb1, rgb2 = self.get_rand_rgb(2)
        _, mask = self.my_set_noise_with_mask(name, rgb1, rgb2, None, fraction)
        return rgb1, rgb2, mask, fraction


    def my_set_texture(self, name, texture_info, is_noise_randomize):
        self.set_checker(name, texture_info["checker_rgb1"], texture_info["checker_rgb2"])
        self.set_gradient(name, texture_info["gradient_rgb1"], texture_info["gradient_rgb2"], texture_info["gradient_vertical"])
        self.set_rgb(name, texture_info["rgb_rgb"])
        if is_noise_randomize:
            self.set_noise(name, texture_info["noise_rgb1"], texture_info["noise_rgb2"], texture_info["noise_fraction"])
        else:
            self.my_set_noise_with_mask(name, texture_info["noise_rgb1"], texture_info["noise_rgb2"], texture_info["noise_mask"], texture_info["noise_fraction"])


    def my_set_noise_with_mask(self, name, rgb1, rgb2, mask=None, fraction=0.9):
        """
        Args:
        - name (str): name of geom
        - rgb1 (array): background color
        - rgb2 (array): color of random noise foreground color
        - fraction (float): fraction of pixels with foreground color
        """
        bitmap = self.get_texture(name).bitmap

        if mask is None:
            h, w = bitmap.shape[:2]
            mask = self.random_state.uniform(size=(h, w)) < fraction

        bitmap[..., :] = np.asarray(rgb1)
        bitmap[mask, :] = np.asarray(rgb2)

        self.upload_texture(name)
        return bitmap, mask







if __name__ == '__main__':
    TextureModder_ = TextureModder()

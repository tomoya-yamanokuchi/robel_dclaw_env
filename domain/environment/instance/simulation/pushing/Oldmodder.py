import numpy as np
from attrdict import AttrDict
from mujoco_py.modder import TextureModder
'''
・mujocoのTextureModderクラスを継承した拡張クラスです
・テクスチャのランダム化の再現性の確保・ステップごとのランダム化の有無を実現するためのクラスです
'''


class myTextureModder(TextureModder):
    def __init__(self, sim, random_state=None):
         super().__init__(sim, random_state)


    def get_rand_texture(self):
        checker  = self.get_rand_checker()   # my original
        gradient = self.get_rand_gradient()  # my original
        rgb      = self.get_rand_rgb()       # mujoco build-in
        noise    = self.get_rand_noise()     # my original
        return AttrDict({
            "checker" : checker,
            "gradient": gradient,
            "rgb"     : rgb,
            "noise"   : noise,
        })


    def get_rand_checker(self):
        rgb1, rgb2 = self.get_rand_rgb(2)
        return AttrDict({
            "rgb1" : rgb1,
            "rgb2" : rgb2,
        })


    def get_rand_gradient(self):
        rgb1, rgb2 = self.get_rand_rgb(2)
        vertical = bool(self.random_state.uniform() > 0.5)
        return AttrDict({
            "rgb1"    : rgb1,
            "rgb2"    : rgb2,
            "vertical": vertical,
        })


    def get_rand_noise(self):
        fraction       = 0.1 + self.random_state.uniform() * 0.8
        rgb1, rgb2     = self.get_rand_rgb(2)
        # ---- mask ----
        something_name = self.sim.model.geom_names[0] # bitmapのshapeを得るためなのでmaterialが定義されているgeomならなんでもいいが，index[0]で常に動くとは限らない
        bitmap         = self.get_texture(something_name).bitmap
        h, w           = bitmap.shape[:2]
        mask           = self.random_state.uniform(size=(h, w)) < fraction
        return AttrDict({
            "fraction": fraction,
            "rgb1"    : rgb1,
            "rgb2"    : rgb2,
            "mask"    : mask,
        })


    def my_set_texture(self, name, texture_info, is_noise_randomize):
        checker  = texture_info.checker
        gradient = texture_info.gradient
        rgb      = texture_info.rgb
        noise    = texture_info.noise
        self.set_checker(name, checker.rgb1, checker.rgb2)
        self.set_gradient(name, gradient.rgb1, gradient.rgb2, gradient.vertical)
        self.set_rgb(name, rgb)
        if is_noise_randomize:
            self.set_noise(name, noise.rgb1, noise.rgb2, noise.fraction)
        else:
            self.my_set_noise_with_mask(name, noise.rgb1, noise.rgb2, noise.mask, noise.fraction)


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

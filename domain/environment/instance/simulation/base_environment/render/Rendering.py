from .texture.TextureFactory import TextureFactory
from .ReturnImage import ReturnImage
from .ImageObject import ImageObject
from .Camera import Camera
from .Light import Light

from .utility import flip_upside_down, reverse_channel



class Rendering:
    def __init__(self,
            sim,
            canonical_rgb,
            config_render,
            config_texture,
            config_camera,
            config_light,
        ):
        self.sim                = sim
        self.canonical_rgb      = canonical_rgb
        self.camera_name_list   = config_render.camera_name_list
        self.width_capture      = config_render.width_capture
        self.height_capture     = config_render.height_capture
        self.texture            = TextureFactory().create(config_texture.randomization_mode)(sim, **config_texture)
        self.camera             = Camera(sim, self.camera_name_list, **config_camera)
        self.light              = Light(sim, **config_light)
        self.ambient            = 0 # ライトの色味の変具合
        self.shadowsize         = 0 # canonicalとrandomizedの色情報を一貫させるため



    def __render_image(self, camera_name):
        img = self.sim.render(
            width       = self.width_capture,
            height      = self.height_capture,
            camera_name = camera_name,
            depth       = False
        )
        img = flip_upside_down(img)
        img = reverse_channel(img)
        return ImageObject(img)


    def _render_canonical(self):
        self.texture.set_canonical_texture(self.canonical_rgb)
        self.texture.set_task_relevant_randomized_texture()
        self.light.set_light_ambient(self.ambient)
        self.light.set_light_castshadow(shadowsize=self.shadowsize)
        self.light.set_light_on()
        return self.__render_image("canonical").channel_last


    def _render_randomized(self):
        self.texture.set_randomized_texture()
        self.light.set_light_ambient(self.ambient)
        self.light.set_light_castshadow(shadowsize=self.shadowsize)
        self.light.set_light_on()
        return self.__render_image("random_nonfix").channel_last


    def render(self):
        image = ReturnImage(
            canonical     = self._render_canonical(),
            # random_nonfix = self._render_randomized(),
            mode          = "step"
        )
        return image









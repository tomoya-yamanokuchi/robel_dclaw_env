from .texture.TextureFactory import TextureFactory
from .RenderImage import RenderImage
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
        self.sim                     = sim
        self.canonical_rgb           = canonical_rgb
        self.force_default_canonical = config_texture.force_default_canonical
        self.camera_name_list        = config_render.camera_name_list
        self.width_capture           = config_render.width_capture
        self.height_capture          = config_render.height_capture
        self.texture                 = TextureFactory().create(config_texture.randomization_mode)(sim, **config_texture)
        self.camera                  = Camera(sim, self.camera_name_list, **config_camera)
        self.light                   = Light(sim, **config_light)
        self.ambient                 = 0 # ライトの色味の変具合
        self.shadowsize              = 0 # canonicalとrandomizedの色情報を一貫させるため



    def __render_image(self, camera_name):
        img_rgb = self.sim.render(
            width       = self.width_capture,
            height      = self.height_capture,
            camera_name = camera_name,
            depth       = False
        )
        img_rgb = flip_upside_down(img_rgb)
        img_bgr = reverse_channel(img_rgb)
        return ImageObject(img_bgr)


    def _render_canonical(self):
        self.texture.set_canonical_texture(self.canonical_rgb)
        if not self.force_default_canonical:
            self.texture.set_task_relevant_randomized_texture()
        self.light.set_light_ambient(self.ambient)
        self.light.set_light_castshadow(shadowsize=self.shadowsize)
        self.light.set_light_on()
        return self.__render_image("canonical")


    def _render_randomized(self):
        self.texture.set_randomized_texture()
        self.light.set_light_ambient(self.ambient)
        self.light.set_light_castshadow(shadowsize=self.shadowsize)
        self.light.set_light_on()
        return self.__render_image("random_nonfix")


    def render(self):
        return RenderImage(
            canonical     = self._render_canonical(),
            # random_nonfix = self._render_randomized(),
        )


    def set_canonical_rgb(self, canonical_rgb):
        self.canonical_rgb = canonical_rgb


    def register_new_randomized_texture_collection(self):
        self.texture.register_new_randomized_texture_collection(include_task_relevant_object=True)






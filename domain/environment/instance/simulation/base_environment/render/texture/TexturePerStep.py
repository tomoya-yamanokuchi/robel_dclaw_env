from pprint import pprint
from .myTextureModder import myTextureModder
from .VisibleGeometry import VisibleGeometry
from .utility import create_texture_collection_without_info



class TexturePerStep:
    def __init__(self,
            sim,
            randomization_mode           : str,
            is_dynamc_noise              : bool,
            task_relevant_geom_group_name: str,
        ):
        self.texture_modder        = myTextureModder(sim)
        self.randomization_mode    = randomization_mode
        self.is_dynamc_noise       = is_dynamc_noise
        self.visible_geometry      = VisibleGeometry(sim, task_relevant_geom_group_name)
        self.grouped_visible_geoms = self.visible_geometry.get_task_relevant_grouped_visible_geometries()
        self.texture_collection    = create_texture_collection_without_info(self.grouped_visible_geoms)


    def register_new_randomized_texture_collection(self, include_task_relevant_object: bool):
        num_assign = self.texture_collection.size()
        if not include_task_relevant_object: num_assign -= 1
        for id in range(num_assign):
            texture_info = self.texture_modder.get_rand_texture()
            self.texture_collection.assign_info_with_id(id=id, info=texture_info)


    def set_canonical_texture(self, rgb_dict: dict):
        self.texture_modder.set_rgb_from_dict(rgb_dict)


    def set_task_relevant_randomized_texture(self):
        assert not self.texture_collection.is_empty_info()

        max_id = max(self.texture_collection.get_id())
        for texture in self.texture_collection.get_textures_from_id(id=max_id):
            '''
                We don't use my_set_texture() function for task-relevant object
                if the domain-dependent information is color, to kepp consistency
                of the domain infomation as much as possible.
            '''
            self.texture_modder.set_rgb(texture.name, texture.info["rgb"])


    def set_randomized_texture(self):
        assert not self.texture_collection.is_empty_info()
        # self.register_new_randomized_texture_collection(task_relevant_object=False)
        # self._set_texture()
        # def _set_texture(self):
        for texture in self.texture_collection.texture:
            self.texture_modder.my_set_texture(
                name               = texture.name,
                texture_info       = texture.info,
                is_noise_randomize = self.is_dynamc_noise
            )

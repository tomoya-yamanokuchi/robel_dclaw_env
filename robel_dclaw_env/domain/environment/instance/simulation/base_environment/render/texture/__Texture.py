from .myTextureModder import myTextureModder
from .TextureObject import TextureObject
from .TextureCollection import TextureCollection
from .VisibleGeometry import VisibleGeometry

class Texture:
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
        self.texture_collection    = self._initialize_task_grouped_texture_collection()
        self.is_texture_randomized = False


    def _initialize_task_grouped_texture_collection(self):
        texture_collection = TextureCollection()
        for id, geom_names in enumerate(self.grouped_visible_geoms):
            for name in geom_names:
                texture = TextureObject(name=name, id=id, info=dict())
                texture_collection.add(texture)
        return texture_collection


    def set_randomized_texture(self):
        if self.randomization_mode == "loaded_static":
           self._set_texture_static_all()

        elif (self.randomization_mode == "per_reset") or (self.randomization_mode == "static"):
            if self.is_texture_randomized is False:
                self._set_texture_rand_all_with_return_info()
                self.is_texture_randomized = True
            self._set_texture_static_all()

        elif self.randomization_mode == "per_step":
            '''
                ・self.texture_collection の状態に注意
                ・ユニークな self.texture_collection を applyすることでtextureを変更している
            '''
            if self.is_texture_randomized is False:
                self._set_texture_rand_all_with_return_info()
                self.is_texture_randomized = True
            self._set_texture_rand_task_irrelevant_with_return_info()
            self._set_texture_static_all()



    def _set_texture_rand_all_with_return_info(self):
        self.texture = {}
        max_id       = max(self.texture_collection.get_id())
        #  系列ごとのランダム化に使用するtextureを作成
        for id in range(max_id+1):
            self.texture[str(id)] = self.texture_modder.get_rand_texture()
        # texture_collectionに作成したtextureの情報を反映させる
        for id in range(max_id+1):
            self.texture_collection.assign_info_with_id(id=id, info=self.texture[str(id)])


    def _set_texture_rand_task_irrelevant_with_return_info(self):
        self.texture = {}
        max_id       = max(self.texture_collection.get_id())
        #  系列ごとのランダム化に使用するtextureを作成
        for id in range(max_id+1):
            if not id == max_id:
                self.texture[str(id)] = self.texture_modder.get_rand_texture()
        # texture_collectionに作成したtextureの情報を反映させる
        for id in range(max_id+1):
            if not id == max_id:
                self.texture_collection.assign_info_with_id(id=id, info=self.texture[str(id)])



    def _set_texture_static_all(self):
        for texture in self.texture_collection.texture:
            self.texture_modder.my_set_texture(texture.name, texture.info, is_noise_randomize=self.is_dynamc_noise)


    def _set_texture_rand_all(self):
        for name in self.visible_geom:
            self.texture_modder.rand_all(name)



    def task_relevant_randomize_texture(self):
        max_id = max(self.texture_collection.get_id()) # 多分バルブ環境でしか動かない（ハードコーディング部分）
        for texture in self.texture_collection.get_textures_from_id(id=max_id):
            # import ipdb; ipdb.set_trace()
            print(texture.info)
            import ipdb; ipdb.set_trace()
            self.texture_modder.my_set_texture(texture.name, texture.info, is_noise_randomize=False)
        import ipdb; ipdb.set_trace()


    def set_rgb(self):
        for texture in self.texture_collection.texture:
            self.texture_modder.set_rgb(texture.name, texture.info["rgb"])



    def set_canonical_texture(self, rgb_dict: dict):
        self.texture_modder.set_rgb_from_dict(rgb_dict)


    def reset_texture_randomization_state(self):
        if self.randomization_mode != "static":
            self.is_texture_randomized = False

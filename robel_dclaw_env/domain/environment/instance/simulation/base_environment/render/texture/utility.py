
from .TextureObject import TextureObject
from .TextureCollection import TextureCollection


def create_texture_collection_without_info(grouped_visible_geoms):
    texture_collection = TextureCollection()
    for id, geom_names in enumerate(grouped_visible_geoms):
        for name in geom_names:
            texture = TextureObject(name=name, id=id, info=dict())
            texture_collection.add(texture)
    return texture_collection



def set_texture_rand_all_with_return_info(texture_collection: TextureCollection):
    assert isinstance(texture_collection, TextureCollection)
    texture = {}
    for id in range(texture_collection.size()):
        texture[str(id)] = self.texture_modder.get_rand_texture()

    # texture_collectionに作成したtextureの情報を反映させる
    for id in range(max_id+1):
        self.texture_collection.assign_info_with_id(id=id, info=self.texture[str(id)])

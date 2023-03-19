from .TextureObject import TextureObject
import numpy as np


class TextureCollection:
    def __init__(self):
        self.texture = []

    def add(self, texture: TextureObject):
        assert isinstance(texture, TextureObject)
        self.texture.append(texture)

    def size(self):
        return len(self.texture)

    def get_id(self):
        '''
        ここで言うidはtask_relevantなgeomのグルーピングを
        するためのものでユニークではない
        '''
        return [texture.id for texture in self.texture]

    def get_name(self):
        return [texture.name for texture in self.texture]

    def get_info(self):
        return [texture.info for texture in self.texture]

    def is_empty_info(self):
        for texture in self.texture:
            if texture.info == {}: return True
        return False

    def assign_info_with_id(self, id: int, info: dict):
        '''
        randomizeするときに便利
        '''
        consistent_texture = list(filter(lambda texture: texture.id==id, self.texture))
        for texture in consistent_texture:
            texture.info = info


    def assign_info_with_name(self, name: int, info: dict):
        '''
        canonicalizeするときに便利
        （人間が設定するときには名前で管理するのが楽なため）
        '''
        consistent_texture = list(filter(lambda texture: name in texture.name, self.texture))
        for texture in consistent_texture:
            texture.info = info

    def get_name_by_id(self, id: int):
        consistent_texture = list(filter(lambda texture: texture.id==id, self.texture))
        return [texture.name for texture in consistent_texture]


    def get_textures_from_id(self, id: int):
        return list(filter(lambda texture: texture.id==id, self.texture))




if __name__ == '__main__':
    import numpy as np

    texture1 = TextureObject(
        name = "valve_x",
        id   = 3,
        info = dict(),
    )

    texture2 = TextureObject(
        name = "roboto_x",
        id   = 1,
        info = dict(),
    )

    texture_collection = TextureCollection()
    texture_collection.add(texture1)
    texture_collection.add(texture2)

    print(texture_collection.get_id())
    print(texture_collection.get_name())
    print(texture_collection.get_info())
    print("-----------------")
    print(texture_collection.assign_info_with_id(id=1, info={"d": 2, "e": 4}))
    print(texture_collection.get_info())

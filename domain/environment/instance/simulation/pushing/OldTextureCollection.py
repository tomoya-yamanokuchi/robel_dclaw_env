from .OldTexture import Texture
import numpy as np


class TextureCollection:
    def __init__(self):
        self.texture = []

    def add(self, texture: Texture):
        assert isinstance(texture, Texture)
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

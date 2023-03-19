from .TexturePerStep import TexturePerStep


class TextureFactory:
    def create(self, name):
        if name == "per_step" : return TexturePerStep

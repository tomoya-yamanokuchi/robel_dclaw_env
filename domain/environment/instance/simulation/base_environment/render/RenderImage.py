from .ImageObject import ImageObject



class RenderImage:
    def __init__(self, **kwargs: dict):
        self.image = {}
        for key, image_object in kwargs.items():
            assert isinstance(image_object, ImageObject)
            self.image[key] = image_object

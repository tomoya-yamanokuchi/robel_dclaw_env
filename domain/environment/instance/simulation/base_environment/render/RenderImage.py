from .ImageObject import ImageObject



class RenderImage:
    def __init__(self, **kwargs: dict):
        self.collection = {}
        for key, image_object in kwargs.items():
            assert isinstance(image_object, ImageObject)
            self.collection[key] = image_object

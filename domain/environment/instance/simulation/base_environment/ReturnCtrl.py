


class ReturnCtrl:
    def __init__(self, **kwargs: dict):
        self.collection = {}
        for key, val in kwargs.items():
            self.collection[key] = val

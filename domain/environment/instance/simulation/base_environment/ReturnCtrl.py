


class ReturnCtrl:
    def __init__(self, **kwargs: dict):
        self.ctrl = {}
        for key, val in kwargs.items():
            self.ctrl[key] = val

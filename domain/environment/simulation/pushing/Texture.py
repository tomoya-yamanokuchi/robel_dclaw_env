from dataclasses import dataclass


@dataclass(frozen=False)
class Texture:
    name      : str
    id        : int
    info      : dict
    def __post_init__(self):
        assert type(self.name) == str
        assert type(self.id)   == int
        assert type(self.info) == dict


if __name__ == '__main__':
    import numpy as np

    texture = Texture(
        name       = "valve_x",
        id         = 3,
         checker   = 3.9,
         gradientnt= 3.9,
         rgb       = 3.9,
         noise     = 3.9,
    )
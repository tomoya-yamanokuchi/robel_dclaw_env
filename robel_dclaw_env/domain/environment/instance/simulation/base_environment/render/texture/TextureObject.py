from dataclasses import dataclass


@dataclass(frozen=False)
class TextureObject:
    name      : str
    id        : int
    info      : dict
    def __post_init__(self):
        assert type(self.name) == str
        assert type(self.id)   == int
        assert type(self.info) == dict


if __name__ == '__main__':
    import numpy as np

    texture = TextureObject(
        name       = "valve_x",
        id         = 3,
        info       = {},
    )

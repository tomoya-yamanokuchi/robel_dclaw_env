from dataclasses import dataclass
import numpy as np


@dataclass(frozen=False)
class ImageObject:
    value: np.ndarray # (w, h, c) shape, rgb image

    def __post_init__(self):
        assert type(self.value) == np.ndarray
        assert self.value.shape[-1] == 3 # check channel last
        shape = self.value.shape
        if len(shape) == 2:
            self.value = self.value[np.newaxis, :, :]
        elif len(shape) !=3:
            raise NotImplementedError()

    @property
    def channel_last(self):
        return self.value

    @property
    def channel_first(self):
        return self.value.transpose(2, 0, 1)



if __name__ == '__main__':
    import numpy as np
    import cv2

    image = np.random.randn(32, 32, 3)
    img   = ImageObject(image)

    img_last  = img.channel_last
    img_first = img.channel_first.transpose(1, 2, 0)
    img_diff  = img_last - img_first
    print(img_diff.sum())

    cv2.imshow("win", np.concatenate((img_last, img_first, img_diff), axis=1))
    cv2.waitKey(5000)

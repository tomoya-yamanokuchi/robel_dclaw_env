import cv2
import time
import mujoco_py
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from ..render.ReturnImage import ReturnImage


class OffscreenViewer:
    def __init__(self, sim):
        self.viewer      = mujoco_py.MjRenderContextOffscreen(sim, 0); time.sleep(1)
        self.window_name = 'offscree_viewer'
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)


    def view(self, image: ReturnImage):
        img_ran  = image.random_nonfix
        img_can  = image.canonical
        img_diff = img_ran - img_can
        cv2.imshow(self.window_name, np.concatenate([img_ran, img_can, img_diff], axis=1))
        cv2.waitKey(50)

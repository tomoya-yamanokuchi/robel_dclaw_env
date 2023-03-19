import cv2
import time
import mujoco_py
import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from ..render.ReturnImage import ReturnImage


class OnscreenViewer:
    def __init__(self, sim):
        self.viewer = mujoco_py.MjViewer(sim); time.sleep(1)


    def view(self, pseud_image):
        self.viewer.render()

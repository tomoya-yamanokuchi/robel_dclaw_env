import os
import pprint

import cv2
import numpy as np
from natsort import natsorted
# ------------------------------------
import sys; import pathlib; p=pathlib.Path("./"); sys.path.append(str(p.parent.resolve()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository

cv2.namedWindow('img', cv2.WINDOW_NORMAL)


repository = Repository(
    dataset_dir  = "./dataset",
    dataset_name = "random_action_claw3_NumSample300_NumColoredNoiseExponent3_2023412214634",
)


for f in repository.get_filenames():
    repository.open(filename=f)
    img_can = repository.repository["image"]["canonical"]
    step, width, height, channel = img_can.shape

    for t in range(step):
        # if t != 0: continue
        # print("({}/{}) [{}] step: {}".format(index+1, num_files, db, t))
        cv2.imshow('img', img_can[t])
        cv2.waitKey(100)

        # print(t)
        # if t==0:
        #     import ipdb; ipdb.set_trace()

    repository.close()

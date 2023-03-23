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
    # dataset_name="dataset_202332322364"
    dataset_name="dataset_2023323223944"
)

db_files  = os.listdir(repository.dataset_save_dir)
db_files  = natsorted(db_files)
num_files = len(db_files)
pprint.pprint(db_files)


for index, db in enumerate(db_files):
    db_name, suffix = db.split(".")
    repository.open(filename=db_name)
    img_can = repository.repository["image"]["canonical"]
    step, width, height, channel = img_can.shape

    for t in range(step):
        print("({}/{}) [{}] st ep: {}".format(index+1, num_files, db, t))
        cv2.imshow('img', img_can[t])
        cv2.waitKey(100)

        # print(t)
        # if t==0:
            # import ipdb; ipdb.set_trace()

    repository.close()

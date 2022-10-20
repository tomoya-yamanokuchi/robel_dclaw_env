import os, pprint
from natsort import natsorted
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
import cv2
cv2.namedWindow('img', cv2.WINDOW_NORMAL)


repository = Repository(
    dataset_dir  = "./dataset",
    dataset_name = "dataset_20221019174710",
)

db_files = os.listdir(repository.dataset_save_dir)
db_files = natsorted(db_files)
# pprint.pprint(db_files)

for db in db_files:
    # import ipdb; ipdb.set_trace()
    db_name, suffix = db.split(".")
    repository.open(filename=db_name)
    image = repository.repository["image"].canonical
    step, width, height, channel = image.shape
    for t in range(step):
        print("[{}] step: {}".format(db, t))
        cv2.imshow('img', image[t])
        cv2.waitKey(10)
        # print(repository.repository["state"].robot_position)

    repository.close()
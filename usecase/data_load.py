import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
import cv2
cv2.namedWindow('img', cv2.WINDOW_NORMAL)


repository = Repository(
    dataset_dir  = "./dataset",
    dataset_name = "dataset_41",
)

num_sequence = 8

for s in range(num_sequence):
    repository.open(filename="action{}_domain0".format(s))
    image = repository.repository["image"].canonical
    step, width, height, channel = image.shape
    for t in range(step):
        print(t)
        cv2.imshow('img', image[t])
        cv2.waitKey(10)

    # print(repository.repository["state"].robot_position)

    repository.close()
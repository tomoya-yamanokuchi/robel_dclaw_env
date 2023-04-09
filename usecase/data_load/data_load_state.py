import os, pprint
import matplotlib
import numpy as np
from natsort import natsorted
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
# import cv2
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


repository = Repository(
    dataset_dir  = "./dataset",
    dataset_name = "nominal_with_noise1_NumSample100_NumColoredNoiseExponent3_20234910122"
)
query_state = "robot_position"

# ------------------------------------------------

state_list = []
for f in repository.get_filenames():
    repository.open(filename=f)
    state = repository.get_state(key=query_state)
    state_list.append(state)
    repository.close()

state_all = np.stack(state_list, axis=0)

# import ipdb; ipdb.set_trace()

plt.plot(state_all[:, :, 0].transpose())
plt.show()

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
    dataset_name = "random_action_claw2_NumSample30_NumColoredNoiseExponent1_202341091956"
)
query_state = "task_space_diff_position"

# ------------------------------------------------

state_list = []
for f in repository.get_filenames():
    repository.open(filename=f)
    # import ipdb; ipdb.set_trace()
    state = repository.repository["ctrl"][query_state]
    state_list.append(state)
    repository.close()

state_all = np.stack(state_list, axis=0)
print("state shape = ", state_all.shape)

# import ipdb; ipdb.set_trace()

plt.plot(state_all[:, :, 1].transpose())
plt.show()

import os, pprint
import matplotlib
import numpy as np
from natsort import natsorted
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from domain.repository.SimulationDataRepository import SimulationDataRepository as Repository
# import cv2
# cv2.namedWindow('img', cv2.WINDOW_NORMAL)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


repository = Repository(
    dataset_dir  = "/nfs/workspace/robel_dclaw_env/dataset",
    # dataset_name = "nominal_with_noise1_NumSample100_NumColoredNoiseExponent3_2023412213522"
    dataset_name="nominal_with_noise1_NumSample100_NumColoredNoiseExponent3_2023421104155"
)
query_state = "object_position"

# ------------------------------------------------

state_list = []
for f in repository.get_filenames():
    repository.open(filename=f)
    # import ipdb; ipdb.set_trace()
    state = repository.repository["state"][query_state]
    state_list.append(state)
    repository.close()

state_all = np.stack(state_list, axis=0)
print("state shape = ", state_all.shape)

# import ipdb; ipdb.set_trace()

# dim_u   = state_all.shape[-1]
# fig, ax = plt.subplots(dim_u, 1, figsize=(4, 6))
# for d in range(dim_u):
#     ax[d].plot(state_all[:, :, d].transpose())
# plt.savefig("./data_load_ctrl.png", dpi=200)

# import ipdb; ipdb.set_trace()
plt.plot(state_all.transpose())
plt.savefig("./data_load_state.png", dpi=200)

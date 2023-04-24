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
    dataset_name="nominal_with_noise0_NumSample10_NumColoredNoiseExponent3_2023422252"
)
# ------------------------------------------------

def plot_ctrl(u, name):
    dim_u   = u.shape[-1]
    fig, ax = plt.subplots(dim_u, 1, figsize=(4, 6))
    for d in range(dim_u):
        ax[d].plot(u[:, :, d].transpose())
        ax[d].set_ylim(-0.5, 0.5)
    plt.savefig("./data_load_ctrl_{}.png".format(name), dpi=200)


def plot_object_position(u, name):
    dim_u   = u.shape[-1]
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.plot(u[:, :].transpose())
    # ax.set_ylim(-0.5, 0.5)
    plt.savefig("./data_load_object_position_{}.png".format(name), dpi=200)


stask_space_diff_position_list = []
joint_space_position_list      = []
object_position_list           = []

for f in repository.get_filenames():
    repository.open(filename=f)
    # import ipdb; ipdb.set_trace()
    ctrl  = repository.repository["ctrl"]
    state = repository.repository["state"]
    stask_space_diff_position_list.append(ctrl["task_space_diff_position"])
    joint_space_position_list.append(ctrl["joint_space_position"])
    object_position_list.append(state["object_position"])
    repository.close()

task_space_diff_position = np.stack(stask_space_diff_position_list, axis=0)
joint_space_position     = np.stack(joint_space_position_list, axis=0)
object_position          = np.stack(object_position_list, axis=0)
print("state shape = ", task_space_diff_position.shape)


plot_ctrl(task_space_diff_position,   name="task_space_diff_position")
plot_ctrl(joint_space_position,       name="joint_space_position")
plot_object_position(object_position, name="object_position")


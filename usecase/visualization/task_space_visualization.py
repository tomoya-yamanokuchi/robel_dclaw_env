import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from custom_service import visualization as vis
from robel_dclaw_env.domain.environment.task_space.manifold_1d.Manifold1D import Manifold1D as TaskSpace
from robel_dclaw_env.domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold as TaskSpaceValueObject
from robel_dclaw_env.custom_service import NTD

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

t = np.linspace(start=0.0, stop=1.0, num=200)
t = t.reshape(-1, 1)
t = np.tile(t, (1, 3))

task_space_position   = TaskSpaceValueObject(NTD(t))
taskspace             = TaskSpace()
end_effector_position = taskspace.task2end(task_space_position)


vis.scatter_3d_color_map(
    x         = end_effector_position.value[0, :, :3],
    cval      = t[:, 0],
    # save_path = "./task_space_Manifold1D.png",
    save_path = None,
)

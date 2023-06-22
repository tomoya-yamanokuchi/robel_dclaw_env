import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from robel_dclaw_env.custom_service import visualization as vis
from robel_dclaw_env.domain.environment.task_space import TaskSpaceBuilder
from torch_numpy_converter import to_numpy, to_tensor, NTD

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt

t = np.linspace(start=0.0, stop=1.0, num=400)
t = t.reshape(-1, 1)
t = np.tile(t, (1, 3))

task_space            = TaskSpaceBuilder().build(env_name="sim_valve", mode="torch")
transformer           = task_space["transformer"]
TaskSpacePosition     = task_space["TaskSpacePosition"]

task_space_position   = TaskSpacePosition(to_tensor(NTD(t)))
end_effector_position = transformer.task2end(task_space_position)

# import ipdb; ipdb.set_trace()
vis.scatter_3d_color_map(
    x         = to_numpy(end_effector_position.value[0, :, 6:]),
    cval      = t[:, 0],
    # save_path = "./task_space_Manifold1D.png",
    save_path = None,
)

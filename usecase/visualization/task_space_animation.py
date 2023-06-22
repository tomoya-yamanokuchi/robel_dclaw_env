import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.custom_service import visualization as vis
from robel_dclaw_env.domain.environment.task_space.manifold_1d.Manifold1D import Manifold1D as TaskSpace


taskspace = TaskSpace()

t = np.linspace(start=0.0, stop=2.0, num=200)
t = t.reshape(-1, 1)
t = np.tile(t, (1, 3))

end = taskspace.task2end(t)

# vis.scatter_3d(end[:, :3])
vis.scatter_3d_animation(end[:, :3], num_history=100, interval=10)

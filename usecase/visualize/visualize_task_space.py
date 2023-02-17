import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from custom_service import visualization as vis
from domain.environment.task_space.TaskSpace import TaskSpace


taskspace = TaskSpace()

t = np.linspace(start=0.0, stop=1.0, num=200)
t = t.reshape(-1, 1)
t = np.tile(t, (1, 3))

end = taskspace.task2end(t)

# import ipdb; ipdb.set_trace()
vis.scatter_3d_color_map(x=end[:, :3], cval=t[:, 0])

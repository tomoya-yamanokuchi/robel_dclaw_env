import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))

from domain.environment.task_space.TaskSpace import TaskSpace
from custom_service import visualization as vis

task_space = TaskSpace()
vis.scatter_3d(task_space.reference_end_effector_position)
vis.scatter_3d_animation(task_space.reference_end_effector_position, num_history=350, interval=30)
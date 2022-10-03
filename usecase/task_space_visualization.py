import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))

from domain.environment.task_space.TaskSpace import TaskSpace
from custom_service import visualization as vis






task_space = TaskSpace()

# import ipdb; ipdb.set_trace()
# vis.scatter_3d(task_space.reference_end_effector_position)
vis.scatter_3d_color_map(
    x          = task_space.reference_end_effector_position,
    cval       = task_space.reference_task_space_position,
    cmap_label = "task space position"
)

# vis.scatter_3d_animation(task_space.reference_end_effector_position, num_history=350, interval=30)






import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.environment.task_space import TaskSpaceBuilder
from robel_dclaw_env.custom_service import visualization as vis



# task_space             = TaskSpaceBuilder().build(env_name="sim_valve", mode="numpy")
task_space             = TaskSpaceBuilder().build(env_name="sim_valve", mode="torch")

task_space_transformer =  task_space["transformer"]

vis.scatter_3d_color_map(
    x          = task_space_transformer.reference_end_effector_position,
    cval       = task_space_transformer.reference_task_space_position,
    cmap_label = "task space position"
)

# vis.scatter_3d_animation(task_space.reference_end_effector_position, num_history=350, interval=30)


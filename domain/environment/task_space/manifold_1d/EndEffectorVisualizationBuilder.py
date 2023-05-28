import numpy as np
from .EndEffectorVisualization import EndEffectorVisualization


class EndEffectorVisualizationBuilder:
    def build(self, end_effector_position, reference_end_effector_position):
        assert reference_end_effector_position.shape == (351, 3)
        vis = EndEffectorVisualization()
        vis.scatter_3d_color_map_reference(
            x         = reference_end_effector_position,
            cval      = np.linspace(start=0.0, stop=1.0, num=reference_end_effector_position.shape[0]),
        )
        vis.scatter_3d_query(
            x = end_effector_position[0, :, :3],
            marker_size = 300,
        )
        vis.show()

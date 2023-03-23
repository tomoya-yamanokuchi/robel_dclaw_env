import sys
import copy
import pathlib
import numpy as np
from pprint import pprint
import mujoco_py
# -------- import from same level directory --------
from .PushingFeedState import PushingFeedState as FeedState
from .PushingReturnState import PushingReturnState as ReturnState
from .PushingReturnCtrl import PushingReturnCtrl as ReturnCtrl
from .CanonicalRGB import CanonicalRGB
# -------- import from upper level directory --------
import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd()))
from domain.environment.instance.simulation.base_environment.BaseEnvironment import BaseEnvironment
from domain.environment.kinematics.ForwardKinematics import ForwardKinematics
from domain.environment.kinematics.InverseKinematics import InverseKinematics
from domain.environment.task_space.end_effector_action_pace.EndEffector2D import EndEffector2D as TaskSpace

from domain.environment.task_space.end_effector_action_pace.TaskSpacePositionValueObject_2D_Plane import TaskSpacePositionValueObject_2D_Plane as TaskSpaceValueObject
from domain.environment.task_space.end_effector_action_pace.EndEffectorPositionValueObject_2D_Plane import EndEffectorPositionValueObject_2D_Plane as EndEffectorValueObject
from custom_service import print_info, NTD

from ..base_environment.SetState import SetState
from ..base_environment.GetState import GetState
from ..base_environment.SetCtrl  import SetCtrl
from ..base_environment.render.Rendering import Rendering
from ..base_environment.viewer.ViewerFactory import ViewerFactory
from ..base_environment.dynamics_parameter.RobotDynamicsParameter import RobotDynamicsParameter
from ..base_environment.joint_range.RobotJointRange import RobotJointRange
from .PushingObjectDyanmicsParameter import PushingObjectDyanmicsParameter
from .PushingObjectJointRange import PushingObjectJointRange
from ..base_environment.ctrl_range.RobotCtrlRange import RobotCtrlRange



class PushingSimulationEnvironment(BaseEnvironment):
    def __init__(self, config, use_render=True):
        super().__init__(config)
        self.config             = config
        self.task_space         = TaskSpace()
        self.use_render         = use_render


    def model_file_reset(self):
        # self._generate_model_file()
        self.model         = self.load_model(self.config.model_file)
        object_geom_names  = [name for name in self.model.geom_names
                if (self.config.texture.task_relevant_geom_group_name in name)]
        self.canonical_rgb = CanonicalRGB(num_object_geom=len(object_geom_names))


    def _generate_model_file(self):
        from domain.xml_generation.ConvexHull2D import ConvexHull2D
        from domain.xml_generation.flattened_2d_meshgrid import flattened_2d_meshgrid
        from domain.xml_generation.PlotConvexHull import PlotConvexHull
        from domain.xml_generation.element_tree.PushingObjectXML import PushingObjectElementTree
        from custom_service import normalize

        convex        = ConvexHull2D(num_sample=7)
        convex_origin = copy.deepcopy(convex)

        # ---------- origin convex ---------
        all_points           = flattened_2d_meshgrid(min=convex.min, max=convex.max, num_points_1axis=30)
        inside_points_origin = convex_origin.get_inside_points(all_points)
        # plot_convex          = PlotConvexHull(convex_origin)
        # plot_convex.plot(all_points, inside_points_origin, "./convex_origin.png")

        # ---------- aligned convex ---------
        convex.hull.points[:, 0] += (inside_points_origin[:, 0].mean())*(-1)
        convex.hull.points[:, 1] += (inside_points_origin[:, 1].mean())*(-1)
        aligned_inside_points = convex.get_inside_points(all_points)
        # plot_convex           = PlotConvexHull(convex)
        # plot_convex.plot(all_points, aligned_inside_points, "./convex_alinged.png")

        # ---------- aligned convex ---------
        nomalized_inside_points  = normalize(aligned_inside_points, x_min=-1.0, x_max=1.0, m=-0.03, M=0.03)
        total_object_body_mass   = 0.05
        individual_geometry_mass = (total_object_body_mass / nomalized_inside_points.shape[0])

        # ---------- generate model file ---------
        pusing_etree = PushingObjectElementTree()
        pusing_etree.add_joint_tree()
        pusing_etree.add_body_tree(nomalized_inside_points, individual_geometry_mass)
        pusing_etree.save_xml()


    def reset(self, state):
        self.model_file_reset()
        if self.sim is not None: return 0
        self.sim = mujoco_py.MjSim(self.model); self.sim.reset()
        RobotDynamicsParameter(self.sim).set(self.config.dynamics.robot)
        PushingObjectDyanmicsParameter(self.sim).set(self.config.dynamics.object)
        RobotJointRange(self.sim).set_range(**self.config.joint_range.robot)
        PushingObjectJointRange(self.sim).set_range(**self.config.joint_range.object)
        RobotCtrlRange(self.sim).set_range()
        self.setState = SetState(self.sim, FeedState,  self.task_space, TaskSpaceValueObject)
        self.getState = GetState(self.sim, FeedState,  self.task_space, EndEffectorValueObject, ReturnState)
        self.setCtrl  = SetCtrl( self.sim, ReturnCtrl, self.task_space, TaskSpaceValueObject)
        self.set_state(state)
        if self.use_render:
            self.viewer    = ViewerFactory().create(self.config.viewer.is_Offscreen)(self.sim)
            self.rendering = Rendering(
                sim            = self.sim,
                canonical_rgb  = self.canonical_rgb.rgb,
                config_render  = self.config.render,
                config_texture = self.config.texture,
                config_camera  = self.config.camera,
                config_light   = self.config.light,
            )
        self.sim.step()


    def render(self):
        assert self.viewer is not None # to be initialized before rendering
        assert self.rendering is not None
        self.image = self.rendering.render()
        return self.image


    def view(self):
        self.viewer.view(self.image)


    def get_state(self):
        return self.getState.get_state()


    def set_state(self, state):
        self.setState.set_state(state)


    def set_ctrl_task_space(self, task_space_abs_ctrl):
        return self.setCtrl.set_ctrl(task_space_abs_ctrl)

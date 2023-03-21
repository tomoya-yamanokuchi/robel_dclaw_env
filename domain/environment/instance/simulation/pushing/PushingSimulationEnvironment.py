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
        self.forward_kinematics = ForwardKinematics()
        self.inverse_kinematics = InverseKinematics()
        self.task_space         = TaskSpace()
        self.use_render         = use_render
        # self.dim_ctrl           = 6 # == dim_task_space_ctrl


    def model_file_reset(self):
        # self._generate_model_file()
        self.model         = self.load_model(self.config.model_file)
        object_geom_names  = [name for name in self.model.geom_names if ("object_geom" in name)]
        self.canonical_rgb = CanonicalRGB(num_object_geom=len(object_geom_names))
        self.sim           = None


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
        # import ipdb; ipdb.set_trace()


    def view(self):
        self.viewer.view(self.image)


    def render(self):
        assert self.viewer is not None # to be initialized before rendering
        assert self.rendering is not None
        self.image = self.rendering.render()
        return self.image


    def set_state(self, state: FeedState):
        assert isinstance(state, FeedState)
        qpos = self._set_qpos(state)
        qvel = self._set_qvel(state)

        # print_info.print_joint_positions(qpos)

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.data.ctrl[:9] = qpos[:9]
        self.sim.data.ctrl[9:] = 0.0
        self.sim.forward()


    def _set_qpos(self, state):
        assert state.task_space_positioin is not None
        qpos = np.zeros(self.sim.model.nq)
        end_effector_position = self.task_space.task2end(TaskSpaceValueObject(NTD(state.task_space_positioin)))
        joint_position        = self.inverse_kinematics.calc(end_effector_position.value.squeeze(0))
        qpos[:9]              = joint_position.squeeze()
        qpos[18:]             = state.object_position # <--- env specific!
        # import ipdb; ipdb.set_trace()
        return qpos


    def _set_qvel(self, state):
        qvel      = np.zeros(self.sim.model.nv)
        qvel[:9]  = state.robot_velocity
        qvel[18:] = state.object_velocity
        return qvel


    def get_state(self):
        state                 = copy.deepcopy(self.sim.get_state())
        robot_position        = state.qpos[:9]
        end_effector_position = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_positioin  = self.task_space.end2task(EndEffectorValueObject(NTD(end_effector_position))).value.squeeze()
        # force                 = self.get_force()
        state = ReturnState(
            robot_position        = robot_position,
            object_position       = state.qpos[18:],
            robot_velocity        = state.qvel[:9],
            object_velocity       = state.qvel[18:],
            end_effector_position = end_effector_position,
            task_space_positioin  = task_space_positioin,
        )
        return state


    def set_ctrl_task_space(self, task_space_abs_ctrl: np.ndarray):
        task_space_position    = TaskSpaceValueObject(NTD(task_space_abs_ctrl))
        ctrl_end_effector      = self.task_space.task2end(task_space_position)                              # 新たな目標値に対応するエンドエフェクタ座標を計算
        ctrl_joint             = self.inverse_kinematics.calc(ctrl_end_effector.value.squeeze(axis=0))      # エンドエフェクタ座標からインバースキネマティクスで関節角度を計算
        self.sim.data.ctrl[:9] = ctrl_joint.squeeze()                                                       # 制御入力としてsimulationで設定
        # ---------------
        dclawCtrl = ReturnCtrl(
            task_space_abs_position  = task_space_abs_ctrl.squeeze(),
            task_space_diff_position = None,
            end_effector_position    = ctrl_end_effector.value.squeeze(),
            joint_space_position     = ctrl_joint.squeeze(),
        )
        return dclawCtrl

import sys
import copy
import pathlib
import numpy as np
from pprint import pprint
import mujoco_py
# -------- import from same level directory --------
from .ValveFeedState import ValveFeedState as FeedState
from .ValveReturnState import ValveReturnState as ReturnState
from .ValveReturnCtrl import ValveReturnCtrl as ReturnCtrl
from .CanonicalRGB import CanonicalRGB
# -------- import from upper level directory --------
import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd()))
from domain.environment.instance.simulation.base_environment.BaseEnvironment import BaseEnvironment
from domain.environment.kinematics.ForwardKinematics import ForwardKinematics
from domain.environment.kinematics.InverseKinematics import InverseKinematics
from domain.environment.task_space.manifold_1d.Manifold1D import Manifold1D as TaskSpace

from domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold as TaskSpaceValueObject
from domain.environment.task_space.manifold_1d.EndEffectorPositionValueObject import EndEffectorPositionValueObject as EndEffectorValueObject
from custom_service import print_info, NTD

from ..base_environment.SetState import SetState
from ..base_environment.GetState import GetState
from ..base_environment.SetCtrl  import SetCtrl
from ..base_environment.render.Rendering import Rendering
from ..base_environment.viewer.ViewerFactory import ViewerFactory
from ..base_environment.dynamics_parameter.RobotDynamicsParameter import RobotDynamicsParameter
from ..base_environment.joint_range.RobotJointRange import RobotJointRange
from .ValveDyanmicsParameter import ValveDyanmicsParameter
from .ValveJointRange import ValveJointRange
from ..base_environment.ctrl_range.RobotCtrlRange import RobotCtrlRange


class ValveSimulationEnvironment(BaseEnvironment):
    def __init__(self, config, use_render=True):
        super().__init__(config)
        self.config             = config
        self.forward_kinematics = ForwardKinematics()
        self.inverse_kinematics = InverseKinematics()
        self.task_space         = TaskSpace()
        self.use_render         = use_render


    def model_file_reset(self):
        # self._generate_model_file()
        self.model         = self.load_model(self.config.model_file)
        self.canonical_rgb = CanonicalRGB()
        self.sim           = None


    def reset(self, state):
        self.model_file_reset()
        if self.sim is not None: return 0
        self.sim = mujoco_py.MjSim(self.model); self.sim.reset()
        RobotDynamicsParameter(self.sim).set(self.config.dynamics.robot)
        ValveDyanmicsParameter(self.sim).set(self.config.dynamics.object)
        RobotJointRange(self.sim).set_range(**self.config.joint_range.robot)
        ValveJointRange(self.sim).set_range(**self.config.joint_range.object)
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


    # def set_target_visible(self):

    #     self._target_bid        = self.model.body_name2id('target')
    #     self._target_sid        = self.model.site_name2id('tmark')
    #     # ------------------------
    #     if self.is_target_visible:
    #         if self.env_name == "blue": self.sim.model.site_rgba[self._target_sid] = [1.,  0.92156863, 0.23137255, 1]
    #         else                      : self.sim.model.site_rgba[self._target_sid] = [0, 0, 1, 1]
    #     else:
    #         self.sim.model.site_rgba[self._target_sid] = [0, 0, 1, 0]



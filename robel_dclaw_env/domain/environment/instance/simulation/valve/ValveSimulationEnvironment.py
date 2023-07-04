import os
import sys
import copy
import pathlib
import numpy as np
from pprint import pprint
import mujoco_py
# -------- import from same level directory --------
from .ValveState import ValveState as State
from .CanonicalRGB import CanonicalRGB
# -------- import from upper level directory --------
import robel_dclaw_env
from robel_dclaw_env.domain.environment.instance.simulation.base_environment.BaseEnvironment import BaseEnvironment
from robel_dclaw_env.domain.environment.kinematics import ForwardKinematics
from robel_dclaw_env.domain.environment.kinematics import InverseKinematics
from task_space import TaskSpaceFactory
from task_space.manifold_1d import TaskSpacePositionValue_1D_Manifold
from ..base_environment.SetState import SetState
from ..base_environment.GetState import GetState
from ..base_environment.SetCtrl  import SetCtrl
from .ValveTarget import ValveTarget
from ..base_environment.render.Rendering import Rendering
from ..base_environment.viewer.ViewerFactory import ViewerFactory
from ..base_environment.dynamics_parameter.RobotDynamicsParameter import RobotDynamicsParameter
from ..base_environment.joint_range.RobotJointRange import RobotJointRange
from .ValveDyanmicsParameter import ValveDyanmicsParameter
from .ValveJointRange import ValveJointRange
from ..base_environment.ctrl_range.RobotCtrlRange import RobotCtrlRange

from robel_dclaw_env.custom_service import to_tensor, NTD


class ValveSimulationEnvironment(BaseEnvironment):
    def __init__(self, config, use_render=True):
        super().__init__(config)
        self.config             = config
        self.forward_kinematics = ForwardKinematics()
        self.inverse_kinematics = InverseKinematics()
        self.use_render         = use_render

        self.env_name               = "sim_valve"
        self.task_space_transformer = TaskSpaceFactory.create_transformer(self.env_name, mode="torch")
        self.TaskSpaceValueObject   = TaskSpaceFactory.create_position(self.env_name)


    def model_file_reset(self):
        absolute_path      = os.path.abspath(robel_dclaw_env.__file__)
        parent_dir         = os.path.dirname(absolute_path)
        model_path         = os.path.join(parent_dir, "domain", "environment", "model", self.config.model_file)
        self.model         = self.load_model(model_path)
        self.canonical_rgb = CanonicalRGB(self.config.xml.rgb.object)
        # import ipdb; ipdb.set_trace()


    def set_object_rgb(self, object_rgb: list):
        self.canonical_rgb = CanonicalRGB(object_rgb)
        self.rendering.set_canonical_rgb(self.canonical_rgb.rgb)


    def reset(self, state, verbose=False):
        if self.sim is None:
            self.model_file_reset()
            self.sim      = mujoco_py.MjSim(self.model)
            self.setState = SetState(self.sim, State, self.task_space_transformer)
            self.getState = GetState(self.sim, State, self.task_space_transformer)
            self.setCtrl  = SetCtrl( self.sim,        self.task_space_transformer)
            self.setTargetPosition = ValveTarget(self.sim)
            self.setTargetPosition.set_target_visible(self.config.target.visible)
            RobotDynamicsParameter(self.sim).set(self.config.dynamics.robot)
            ValveDyanmicsParameter(self.sim).set(self.config.dynamics.object)
            RobotJointRange(self.sim).set_range(**self.config.joint_range.robot)
            ValveJointRange(self.sim).set_range(**self.config.joint_range.object)
            RobotCtrlRange(self.sim).set_range()
            self._initialize_viewer_and_render()
        self.sim.reset()
        self.set_state(state)
        if self.use_render:
            self.rendering.register_new_randomized_texture_collection()
        # print(self.sim.get_state().time)
        self.sim.step()
        if verbose: print("\n <---- reset --->\n")


    def _initialize_viewer_and_render(self):
        if not self.use_render: return
        if self.viewer is None:
            self.viewer = ViewerFactory().create(self.config.viewer.is_Offscreen)(self.sim)
        if self.rendering is None:
            self.rendering = Rendering(
                sim            = self.sim,
                canonical_rgb  = self.canonical_rgb.rgb,
                config_render  = self.config.render,
                config_texture = self.config.texture,
                config_camera  = self.config.camera,
                config_light   = self.config.light,
            )


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


    def set_ctrl_task_space(self, TaskSpacePosition: TaskSpacePositionValue_1D_Manifold):
        return self.setCtrl.set_ctrl(TaskSpacePosition)


    def set_ctrl_joint_space_position(self, joint_space_position):
        return self.setCtrl.set_ctrl_joint_space_position(joint_space_position)


    def set_target_position(self, target_position):
        self.setTargetPosition.set_target_position(target_position)

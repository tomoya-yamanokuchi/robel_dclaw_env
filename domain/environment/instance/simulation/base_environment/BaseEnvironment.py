import sys
import cv2
import numpy as np
import time
from typing import List
from pprint import pprint
import copy
import mujoco_py

# -------- import from service --------
from custom_service import dictionary_operation as dictOps
# -------- import from same level directory --------

# from .Texture import Texture
# from .TextureCollection import TextureCollection
# from .image.ReturnImage import Image
# from .image.ImageObject import ImageObject
from .AbstractEnvironment import AbstractEnvironment
from .render.Rendering import Rendering
# from .viewer.OffscreenViewer import Viewer
# from .Light import Light
# from .Camera import Camera
# -------- import from upper level directory --------
import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd()))



class BaseEnvironment(AbstractEnvironment):
    def __init__(self, config):
        self.config            = config
        self.inplicit_step     = config.inplicit_step
        self.env_name          = config.env_name
        self.env_color         = config.env_color
        self.claw_jnt_range_lb = config.claw_jnt_range_lb
        self.claw_jnt_range_ub = config.claw_jnt_range_ub
        self.dynamics          = config.dynamics
        self.verbose           = config.verbose
        self._target_position  = None
        self.sim               = None
        self.camera_modder     = None

        # -----------------
        self.viewer = None


    def load_model(self, model_file):
        repository_name  = "robel-dclaw-env"
        sys_path_leaf    = [path.split("/")[-1] for path in sys.path]   # 全てのパスの末端ディレクトリを取得
        assert repository_name in sys_path_leaf                         # 末端ディレクトリにリポジトリ名が含まれているか確認
        index_model_path = sys_path_leaf.index(repository_name)         # リポジトリがあるパスを抽出
        xml_path         = "{}/domain/environment/model/{}".format(sys.path[index_model_path], model_file)
        # import ipdb; ipdb.set_trace()
        return mujoco_py.load_model_from_path(xml_path)



    def _set_robot_dynamics_parameter(self, randparams_dict: dict) -> None:
        set_dynamics_parameter_function = {
            "kp_claw"            :  self.__set_claw_actuator_gain_position,
            "damping_claw"       :  self.__set_claw_damping,
            "frictionloss_claw"  :  self.__set_claw_frictionloss,
        }
        for key, value in randparams_dict.items():
            set_dynamics_parameter_function[key](value)


    def __set_claw_actuator_gain_position(self, kp):
        self.sim.model.actuator_gainprm[:9, 0] =  kp
        self.sim.model.actuator_biasprm[:9, 1] = -kp


    def __set_claw_damping(self, value):
        self.sim.model.dof_damping[:9] = value


    def __set_claw_frictionloss(self, value):
        self.sim.model.dof_frictionloss[:9] = value


    # def create_mujoco_related_instance(self):
    #     if self.sim is not None: return 0
    #     if self.verbose: print("\n << create_mujoco_related_instance >> \n")
    #     self.sim             = mujoco_py.MjSim(self.model)
        # self.rendering       = Rendering(self.sim, self.canonical_rgb_dict **self.config.render, **self.config.light)
        # self.texture_modder  = TextureModder(self.sim)
        # self.camera = Camera(self.sim, self.config.camera_name_list, **self.config.camera)
        # self.camera.set_camera_posture()
        # self.light  = Light(self.sim, **self.config.light)
        # self.__createTexutureCollection()



    def set_environment_parameters(self, _set_object_dynamics_parameter):
        self._set_robot_dynamics_parameter(self.dynamics.robot)
        _set_object_dynamics_parameter(self.dynamics.object)
        # self._set_camera_position(self.camera)




    def set_target_position(self, target_position):
        '''
            ・バルブの目標状態の値をセット
            ・target_position: 1次元の数値
            ・renderするときに _target_position が None でなければ描画されます
        '''
        target_position       = float(target_position)
        self._target_position = target_position
        self.sim.model.body_quat[self._target_bid] = euler2quat(0, 0, float(self._target_position))


    def set_ctrl_joint(self, ctrl):
        assert ctrl.shape == (9,), '[expected: {0}, input: {1}]'.format((9,), ctrl.shape)
        self.sim.data.ctrl[:9] = ctrl


    def _step_with_inplicit_step(self, is_view):
        '''
        ・一回の sim.step() では，制御入力で与えた目標位置まで到達しないため，これを避けたい時に使います
        ・sim-to-realでは1ステップの状態遷移の違いがそのままダイナミクスのreality-gapとなるため，
        　これを避けるために複数回の sim.step() を内包する当該関数を作成してあります
        '''
        for i in range(self.inplicit_step):
            self.sim.step()
            if is_view: self.view()

    def step(self, is_view=False):
        self._step_with_inplicit_step(is_view)


    @property
    def ctrl(self):
        return self.sim.data.ctrl

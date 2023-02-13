import sys
import pathlib

p = pathlib.Path()
sys.path.append(str(p.cwd()))

import math
from traceback import print_tb
import cv2
import os
from matplotlib import ticker
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import pickle
import pprint
from typing import List
import copy
import mujoco_py
from mujoco_py.modder import LightModder, CameraModder
from .my_mujoco.modder import myTextureModder as TextureModder
from numpy.lib.function_base import append
from transforms3d.euler import euler2quat, quat2euler

# 同階層ディレクトリからのインポート
from .DclawEnvironmentRGBFactory import DclawEnvironmentRGBFactory

# 上位ディレクトリからのインポート
p_file = pathlib.Path(__file__)
path_environment = "/".join(str(p_file).split("/")[:-2])
sys.path.append(path_environment)
from ..DClawState import DClawState
from ..AbstractEnvironment import AbstractEnvironment
from ....custom_service.ImageObject import ImageObject
from ....custom_service import dictionary_operation as dictOps
from ..kinematics.ForwardKinematics import ForwardKinematics
from ..kinematics.InverseKinematics import InverseKinematics
from ..task_space.TaskSpace import TaskSpace



class DClawSimulationEnvironmentOptoForce(AbstractEnvironment):
    def __init__(self, config):
        self.width_capture               = config.width_capture
        self.height_capture              = config.height_capture
        self.camera_name_list            = config.camera_name_list
        self.inplicit_step               = config.inplicit_step
        self.env_name                    = config.env_name
        self.env_color                   = config.env_color
        self.claw_jnt_range_lb           = config.claw_jnt_range_lb
        self.claw_jnt_range_ub           = config.claw_jnt_range_ub
        self.valve_jnt_range_lb          = config.object_jnt_range_lb
        self.valve_jnt_range_ub          = config.object_jnt_range_ub
        # self.is_use_render               = config.is_use_render
        self.is_Offscreen                = config.is_Offscreen
        self.is_target_visible           = config.is_target_visible
        self.model                       = self.load_model(config.model_file)
        self.light_index_list            = [i for i in config.light.values()]
        self.randomize_texture_mode      = config.randomize_texture_mode
        self.is_noise_randomize_per_step = config.is_noise_randomize_per_step
        self.dynamics                    = config.dynamics
        self.camera                      = config.camera
        self.light                       = config.light

        self._valve_jnt_id               = self.model.joint_name2id('valve_OBJRx')
        self._target_bid                 = self.model.body_name2id('target')
        self._target_sid                 = self.model.site_name2id('tmark')

        self.optoforce_geom_id_dict = {
            "claw1" : self.model.geom_name2id('FFL12_phy_optoforce'),
            "claw2" : self.model.geom_name2id('MFL22_phy_optoforce'),
            "claw3" : self.model.geom_name2id('THL32_phy_optoforce'),
        }
        self.index_claw = {
            "claw1" : [0, 1, 2],
            "claw2" : [3, 4, 5],
            "claw3" : [6, 7, 8],
        }

        self._target_position            = None
        self.sim                         = None
        self.viewer                      = None
        self.texture_modder              = None
        self.camera_modder               = None

        self.forward_kinematics          = ForwardKinematics()
        self.inverse_kinematics          = InverseKinematics()
        self.task_space                  = TaskSpace()



    def load_model(self, model_file):
        repository_name  = "robel-dclaw-env"
        sys_path_leaf    = [path.split("/")[-1] for path in sys.path]   # 全てのパスの末端ディレクトリを取得
        assert repository_name in sys_path_leaf                         # 末端ディレクトリにリポジトリ名が含まれているか確認
        index_model_path = sys_path_leaf.index(repository_name)         # リポジトリがあるパスを抽出
        xml_path         = "{}/domain/environment/model/{}".format(sys.path[index_model_path], model_file)
        return mujoco_py.load_model_from_path(xml_path)


    def _set_geom_names_randomize_target(self):
        self.visible_geom_group = [0, 1, 2] # XMLファイルと整合性が取れるようにする
        self.geom_names_randomize_target = []
        for name in self.sim.model.geom_names:
            id    = self.sim.model.geom_name2id(name)
            group = self.sim.model.geom_group[id]
            if group in self.visible_geom_group:
                self.geom_names_randomize_target.append(name)


    def view(self):
        if self.is_Offscreen:
            cv2.imshow(self.cv2_window_name, np.concatenate([img.channel_last for img in self.img_dict.values()], axis=1))
            cv2.waitKey(50)
        else:
            self.viewer.render()


    def _flip(self, img):
        return img[::-1].copy()


    def _reverse_channel(self, img):
        # for convert Color BGR2RGB
        return img[:,:,::-1].copy()


    def _render_and_convert_color(self, camera_name):
        # if self._target_position is not None:
            # self.sim.model.body_quat[self._target_bid] = euler2quat(0, 0, float(self._target_position))
        img = self.sim.render(width=self.width_capture, height=self.height_capture, camera_name=camera_name, depth=False)
        img = self._flip(img)
        img = self._reverse_channel(img)
        return ImageObject(img)


    def set_light_on(self, use_light_index_list):
        self.model.light_active[:] = 0
        for i in use_light_index_list:
            self.model.light_active[i] = 1

    def _set_texture_rand_all_with_return_info(self):
        self.texture = {}
        for name in self.geom_names_randomize_target:
            self.texture[name] = self.texture_modder.my_rand_all(name)

    def _set_texture_static_all(self):
        for geom_name, texture in self.texture.items():
            self.texture_modder.my_set_texture(geom_name, texture, is_noise_randomize=self.is_noise_randomize_per_step)

    def _set_texture_rand_all(self):
        for name in self.geom_names_randomize_target:
            self.texture_modder.rand_all(name)


    def randomize_texture(self):
        if self.randomize_texture_mode == "loaded_static":
           self._set_texture_static_all()

        elif self.randomize_texture_mode == "per_reset":
            if self.is_texture_randomized is False:
                # print(self.is_texture_randomized)
                self._set_texture_rand_all_with_return_info()
                self.is_texture_randomized = True
            self._set_texture_static_all()

        elif self.randomize_texture_mode == "per_step":
            # print("ddddd")
            self._set_texture_rand_all()
            self.is_texture_randomized = True


    def canonicalize_texture(self):
        for name in self.geom_names_randomize_target:
            # print("name : {}, rgb {}".format(name, self.rgb[name]))
            self.texture_modder.set_rgb(name, self.rgb[name])


    def _render(self, camera_name: str="canonical"):
        if   "canonical" in camera_name: shadowsize = 0;    self.canonicalize_texture()
        elif    "random" in camera_name: shadowsize = 1024; self.randomize_texture()
        elif  "overview" in camera_name: shadowsize = 0;    self.randomize_texture()
        else                           : raise NotImplementedError()
        self.set_light_castshadow(shadowsize=shadowsize)
        self.set_light_on(self.light_index_list)
        return self._render_and_convert_color(camera_name)


    def set_light_castshadow(self, shadowsize):
        self.model.vis.quality.shadowsize = shadowsize
        is_castshadow = 0 if shadowsize==0 else 1
        for name in self.model.light_names:
            self.light_modder.set_castshadow(name, is_castshadow)


    def render(self, camera_name_list: str=None, iteration: int=1):
        assert self.is_Offscreen is True, "Please set is_Offscreen = True"
        if camera_name_list is None:
            camera_name_list = self.camera_name_list

        img_dict = {}
        for i in range(iteration):
            for camera_name in camera_name_list:
                img_dict[camera_name] = self._render(camera_name)
        self.img_dict = img_dict
        return copy.deepcopy(self.img_dict)


    def check_camera_pos(self):
        self.sim.reset()
        for i in range(100):
            diff = 0.1
            self.model.cam_pos[0][0] = self.rs.uniform(self.default_cam_pos[0][0] - diff, self.default_cam_pos[0][0] + diff)  # x-axis
            self.model.cam_pos[0][1] = self.rs.uniform(self.default_cam_pos[0][1] - diff, self.default_cam_pos[0][1] + diff)  # x-axis
            self.model.cam_pos[0][2] = self.rs.uniform(self.default_cam_pos[0][2] - diff, self.default_cam_pos[0][2] + diff)  # z-axis
            self.sim.step()
            self.render()


    def set_camera_position(self, camera_parameter: dict):
        pos    = [0]*3
        pos[0] = camera_parameter["x_coordinate"]
        pos[1] = camera_parameter["y_coordinate"]
        pos[2] = camera_parameter["z_distance"]

        orientation = camera_parameter["orientation"]
        quat        = euler2quat(np.deg2rad(orientation), 0.0, np.pi/2)

        reset_cam_names = [cam_name for cam_name in self.sim.model.camera_names if "nonfix" in cam_name]
        for cam_name in reset_cam_names:
            self.camera_modder.set_pos(cam_name, pos)
            self.camera_modder.set_quat(cam_name, quat)
        self.sim.step()
        return pos, orientation


    def set_camera_position_with_all_euler(self, camera_parameter: dict):
        pos    = [0]*3
        pos[0] = camera_parameter["x_coordinate"]
        pos[1] = camera_parameter["y_coordinate"]
        pos[2] = camera_parameter["z_distance"]

        euler = camera_parameter["euler"]
        quat        = euler2quat(*np.deg2rad(euler))

        reset_cam_names = ["cam_canonical_pos_nonfix", "cam_random_pos_nonfix"]
        for cam_name in reset_cam_names:
            self.camera_modder.set_pos(cam_name, pos)
            self.camera_modder.set_quat(cam_name, quat)
        self.sim.step()
        return pos, euler



    def set_light_position(self, light_position: dict):
        assert len(light_position.keys()) == 2
        self.use_light_index_list_random = [int(light_position["light1"]), int(light_position["light2"])]



    def get_camera_parameter(self, isDict: bool = False):
        (x, y, z)  = self.camera_modder.get_pos(name="cam_canonical_pos_nonfix")
        quat       = self.camera_modder.get_quat(name="cam_canonical_pos_nonfix")
        params     = {
                "x_coordinate": x,
                "y_coordinate": y,
                "z_distance"  : z,
                "orientation" : quat2euler(quat)[0]
        }
        if isDict is False:
            params = dictOps.dict2numpyarray(params)
        return params


    def get_light_parameter(self, isDict: bool = False):
        assert len(self.use_light_index_list_random) == 2
        params     = {
            "light1" : self.use_light_index_list_random[0],
            "light2" : self.use_light_index_list_random[1],
        }
        if isDict is False:
            params = dictOps.dict2numpyarray(params)
        return params


    def get_state(self):
        env_state             = copy.deepcopy(self.sim.get_state())
        robot_position        = env_state.qpos[:9]
        end_effector_position = self.forward_kinematics.calc(robot_position)
        force                 = self.get_force()
        state = DClawState(
            robot_position        = robot_position,
            object_position       = env_state.qpos[-1:],
            robot_velocity        = env_state.qvel[:9],
            object_velocity       = env_state.qvel[-1:],
            force                 = force,
            end_effector_position = end_effector_position[0],
        )
        return state


    def get_force(self):
        num_data_per_force_sensor = 3 # (Fx, Fy, Fz)
        force = np.zeros(self.num_force_sensor * num_data_per_force_sensor)
        for i in range(self.sim.data.ncon):
            con = self.sim.data.contact[i]
            for geom in [con.geom1, con.geom2]:
                if geom in list(self.optoforce_geom_id_dict.values()):
                    contact_claw = [k for k, v in self.optoforce_geom_id_dict.items() if v == geom]
                    assert len(contact_claw) == 1
                    force[self.index_claw[contact_claw[0]]] = np.take(self.sim.data.sensordata, self.index_claw[contact_claw[0]])
        # print("force: [{: .3f} {: .3f} {: .3f}], [{: .3f} {: .3f} {: .3f}], [{: .3f} {: .3f} {: .3f}]".format(
        #     force[0], force[1], force[2],     force[3], force[4], force[5],     force[6], force[7], force[8]))
        return force


    def set_state(self, qpos, qvel, sensordata):
        assert(      qpos.shape == self.model.nq,          "      qpos.shape {} != self.model.nq {}".format(qpos.shape, self.model.nq))
        assert(      qvel.shape == self.model.nv,          "      qvel.shape {} != self.model.nv {}".format(qvel.shape, self.model.nv))
        assert(sensordata.shape == self.model.nsensordata, "sensordata.shape {} != self.model.nsensordata {}".format(sensordata.shape, self.model.nsensordata))

        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.data.ctrl[:9]      = qpos[:9]
        self.sim.data.ctrl[9:]      = 0.0
        self.sim.data.sensordata[:] = sensordata
        self.sim.forward()


    def set_dynamics_parameter(self, randparams_dict: dict) -> None:
        set_dynamics_parameter_function = {
            "kp_claw"            :  self.set_claw_actuator_gain_position,
            "damping_claw"       :  self.set_claw_damping,
            "frictionloss_claw"  :  self.set_claw_frictionloss,
            "kp_valve"           :  self.set_valve_actuator_gain_position,
            "kv_valve"           :  self.set_valve_actuator_gain_velocity,
            "damping_valve"      :  self.set_valve_damping,
            "frictionloss_valve" :  self.set_valve_frictionloss,
        }
        for key, value in randparams_dict.items():
            set_dynamics_parameter_function[key](value)


    def set_claw_actuator_gain_position(self, kp):
        self.sim.model.actuator_gainprm[:9, 0] =  kp
        self.sim.model.actuator_biasprm[:9, 1] = -kp


    def set_claw_damping(self, value):
        self.sim.model.dof_damping[:9] = value


    def set_claw_frictionloss(self, value):
        self.sim.model.dof_frictionloss[:9] = value


    def set_valve_actuator_gain_position(self, kp):
        self.sim.model.actuator_gainprm[-2, 0] =  kp
        self.sim.model.actuator_biasprm[-2, 1] = -kp


    def set_valve_actuator_gain_velocity(self, kv):
        self.sim.model.actuator_gainprm[-1, 0] =  kv
        self.sim.model.actuator_biasprm[-1, 2] = -kv


    def set_valve_damping(self, value):
        self.sim.model.dof_damping[-1] = value


    def set_valve_frictionloss(self, value):
        self.sim.model.dof_frictionloss[-1] = value


    def get_dynamics_parameter(self, isDict: bool = False):
        assert type(isDict) == bool
        dynamics_parameter = {
                "kp_claw"            : self.sim.model.actuator_gainprm[0, 0],
                "damping_claw"       : self.sim.model.dof_damping[0],
                "frictionloss_claw"  : self.sim.model.dof_frictionloss[0],
                "kp_valve"           : self.sim.model.actuator_gainprm[-2, 0],
                "kv_valve"           : self.sim.model.actuator_gainprm[-1, 0],
                "damping_valve"      : self.sim.model.dof_damping[-1],
                "frictionloss_valve" : self.sim.model.dof_frictionloss[-1]
        }
        if isDict is False:
            dynamics_parameter = dictOps.dict2numpyarray(dynamics_parameter)
        return dynamics_parameter


    def _create_qpos_qvel_from_InitialState(self, DClawState_: DClawState):
        # qpos
        qpos       = np.zeros(self.sim.model.nq)
        qpos[:9]   = DClawState_.robot_position
        qpos[-1]   = DClawState_.object_position
        # qvel
        qvel       = np.zeros(self.sim.model.nq)
        qvel[:9]   = DClawState_.robot_velocity
        qvel[-1]   = DClawState_.object_velocity

        sensordata = DClawState_.force
        return qpos, qvel, sensordata


    def reset(self, DClawState_: DClawState):
        assert isinstance(DClawState_, DClawState)
        self.is_texture_randomized = False
        if self.sim is None:
            self._create_mujoco_related_instance()
        self.sim.reset()
        self.set_jnt_range()
        self.set_ctrl_range()
        self.set_dynamics_parameter(self.dynamics)
        self.set_camera_position(self.camera)
        self.set_light_position(self.light)
        self.set_target_visible(self.is_target_visible)
        qpos, qvel, sensordata = self._create_qpos_qvel_from_InitialState(DClawState_)
        self.set_state(qpos=qpos, qvel=qvel, sensordata=sensordata)
        if self.is_Offscreen: self.render(iteration=1)
        self.sim.step()



    def _create_mujoco_related_instance(self):
        self.sim = mujoco_py.MjSim(self.model)

        self.num_force_sensor = len(self.sim.model.sensor_names)
        assert self.num_force_sensor == 3

        if self.is_Offscreen:
            self.viewer          = mujoco_py.MjRenderContextOffscreen(self.sim, 0)
            self.cv2_window_name = 'viewer'
            cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)
            print(" init --> viewer"); time.sleep(2)
        else:
            self.viewer = mujoco_py.MjViewer(self.sim)
            print(" init --> "); time.sleep(2)

        # self.viewer.vopt.flags[mujoco_py.const.VIS_CONTACTFORCE] = 1 # force sensorではないbuild-inのcontactを可視化
        # -------------------
        self.texture_modder  = TextureModder(self.sim);                                         print(" init --> texture_modder")
        self.camera_modder   = CameraModder(self.sim);                                          print(" init --> camera_modder")
        self.light_modder    = LightModder(self.sim);                                           print(" init --> light_modder")
        self.default_cam_pos = self.camera_modder.get_pos("canonical");                         print(" init --> default_cam_pos")
        self._set_geom_names_randomize_target();                                                print(" init --> _set_geom_names_randomize_target()")
        factory              = DclawEnvironmentRGBFactory(self.geom_names_randomize_target);    print(" init --> factory")
        self.rgb             = factory.create(self.env_color);                                  print(" init --> self.rgb")




    def set_target_position(self, target_position):
        '''
            ・バルブの目標状態の値をセット
            ・target_position: 1次元の数値
            ・renderするときに _target_position が None でなければ描画されます
        '''
        target_position       = float(target_position)
        self._target_position = target_position
        self.sim.model.body_quat[self._target_bid] = euler2quat(0, 0, float(self._target_position))


    def set_target_visible(self, is_visible):
        if is_visible:
            if self.env_name == "blue": self.sim.model.site_rgba[self._target_sid] = [1.,  0.92156863, 0.23137255, 1]
            else                      : self.sim.model.site_rgba[self._target_sid] = [0, 0, 1, 1]
        else:
            self.sim.model.site_rgba[self._target_sid] = [0, 0, 1, 0]


    def set_ctrl_joint(self, ctrl):
        assert ctrl.shape == (9,), '[expected: {0}, input: {1}]'.format((9,), ctrl.shape)
        self.sim.data.ctrl[:9] = ctrl


    def set_ctrl_task(self, ctrl):
        assert ctrl.shape == (3,), '[expected: {0}, input: {1}]'.format((3,), ctrl.shape)
        ctrl_end_effector = self.task_space.calc(ctrl)
        ctrl_joint        = self.inverse_kinematics.calc(ctrl_end_effector)
        self.sim.data.ctrl[:9] = ctrl_joint.squeeze()


    def set_jnt_range(self):
        claw_jnt_range_num = len(self.claw_jnt_range_ub)
        # --- claw ---
        jnt_index = 0
        if claw_jnt_range_num == 3:
            for i in range(3):
                for k in range(3):
                    self.sim.model.jnt_range[jnt_index, 0] = self.claw_jnt_range_lb[k]
                    self.sim.model.jnt_range[jnt_index, 1] = self.claw_jnt_range_ub[k]
                    jnt_index += 1
        elif claw_jnt_range_num == 9:
            for jnt_index in range(9):
                self.sim.model.jnt_range[jnt_index, 0] = self.claw_jnt_range_lb[jnt_index]
                self.sim.model.jnt_range[jnt_index, 1] = self.claw_jnt_range_ub[jnt_index]
        else:
            raise NotImplementedError()

        # --- valve ---
        self.sim.model.jnt_range[self._valve_jnt_id, 0] = self.valve_jnt_range_lb
        self.sim.model.jnt_range[self._valve_jnt_id, 1] = self.valve_jnt_range_ub


    def set_ctrl_range(self):
        claw_index = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
        for claw_index_unit in claw_index:
            self.sim.model.actuator_ctrlrange[claw_index_unit[0], 0] = np.deg2rad(-90)
            self.sim.model.actuator_ctrlrange[claw_index_unit[0], 1] = np.deg2rad(90)

            self.sim.model.actuator_ctrlrange[claw_index_unit[1], 0] = np.deg2rad(-90)
            self.sim.model.actuator_ctrlrange[claw_index_unit[1], 1] = np.deg2rad(90)

            self.sim.model.actuator_ctrlrange[claw_index_unit[2], 0] = np.deg2rad(-90)
            self.sim.model.actuator_ctrlrange[claw_index_unit[2], 1] = np.deg2rad(90)
        return 0


    def _step_with_inplicit_step(self):
        '''
        ・一回の sim.step() では，制御入力で与えた目標位置まで到達しないため，これを避けたい時に使います
        ・sim-to-realでは1ステップの状態遷移の違いがそのままダイナミクスのreality-gapとなるため，
        　これを避けるために複数回の sim.step() を内包する当該関数を作成してあります
        ・必要ない場合には def step(self) 関数を使用して下さい
        '''
        for i in range(self.inplicit_step):
            self.sim.step()


    def step(self):
        self._step_with_inplicit_step()
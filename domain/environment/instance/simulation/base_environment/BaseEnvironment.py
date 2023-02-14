import sys
import cv2
import numpy as np
import time
from typing import List
from pprint import pprint
import copy
import mujoco_py
from mujoco_py.modder import LightModder, CameraModder
from transforms3d.euler import euler2quat, quat2euler
# -------- import from service --------
from custom_service import dictionary_operation as dictOps
# -------- import from same level directory --------
from .my_mujoco.modder import myTextureModder as TextureModder
from .Texture import Texture
from .TextureCollection import TextureCollection
from ....Image import Image
from .image.ImageObject import ImageObject
from .AbstractEnvironment import AbstractEnvironment
# -------- import from upper level directory --------
import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd()))



class BaseEnvironment(AbstractEnvironment):
    def __init__(self, config):
        self.width_capture                 = config.width_capture
        self.height_capture                = config.height_capture
        self.camera_name_list              = config.camera_name_list
        self.inplicit_step                 = config.inplicit_step
        self.env_name                      = config.env_name
        self.env_color                     = config.env_color
        self.claw_jnt_range_lb             = config.claw_jnt_range_lb
        self.claw_jnt_range_ub             = config.claw_jnt_range_ub
        self.valve_jnt_range_lb            = config.object_jnt_range_lb
        self.valve_jnt_range_ub            = config.object_jnt_range_ub
        self.is_Offscreen                  = config.is_Offscreen
        self.is_target_visible             = config.is_target_visible
        self.model                         = self.load_model(config.model_file)
        self.light_index_list              = [i for i in config.light.values()]
        self.randomize_texture_mode        = config.randomize_texture_mode
        self.is_noise_randomize_per_step   = config.is_noise_randomize_per_step
        self.task_relevant_geom_group_name = config.task_relevant_geom_group_name
        self.dynamics                      = config.dynamics
        self.camera                        = config.camera
        self.light                         = config.light

        self._valve_jnt_id                 = self.model.joint_name2id('valve_OBJRx')
        self._target_bid                   = self.model.body_name2id('target')
        self._target_sid                   = self.model.site_name2id('tmark')

        self._target_position              = None
        self.sim                           = None
        self.viewer                        = None
        self.texture_modder                = None
        self.camera_modder                 = None
        self.is_texture_randomized         = False


    def load_model(self, model_file):
        repository_name  = "robel-dclaw-env"
        sys_path_leaf    = [path.split("/")[-1] for path in sys.path]   # 全てのパスの末端ディレクトリを取得
        assert repository_name in sys_path_leaf                         # 末端ディレクトリにリポジトリ名が含まれているか確認
        index_model_path = sys_path_leaf.index(repository_name)         # リポジトリがあるパスを抽出
        xml_path         = "{}/domain/environment/model/{}".format(sys.path[index_model_path], model_file)
        # import ipdb; ipdb.set_trace()
        return mujoco_py.load_model_from_path(xml_path)


    def __createTexutureCollection(self):
        # ------ 可視化に関係しているgeomだけを抽出する ------
        self.visible_geom_group = [0, 1, 2] # XMLファイル内でのgroup番号と整合性が取れるように設定する
        self.visible_geom = []
        for name in self.sim.model.geom_names:
            id    = self.sim.model.geom_name2id(name)
            group = self.sim.model.geom_group[id]
            if group in self.visible_geom_group:
                self.visible_geom.append(name)

        # import ipdb; ipdb.set_trace()
        # ------ バルブの色を３つ揃えるための工夫 ------
        task_relevant_geoms = [x for x in self.visible_geom if self.task_relevant_geom_group_name in x] # タスク関連のgeomを抽出

        [self.visible_geom.remove(name) for name in task_relevant_geoms] # 抽出したgeomをもとのlistから削除

        # タスク関連のgeomとデータ形式を合わせるためもとのlist[str]の各要素をlistで包む
        for i in range(len(self.visible_geom)):
            self.visible_geom[i] = [self.visible_geom[i]]

        # タスク関連geomをその他のgeomと結合させる
        self.visible_geom.append(task_relevant_geoms)

        # テクスチャのコレクションを作成
        self.texture_collection = TextureCollection()
        for id, geom_names in enumerate(self.visible_geom):
            for name in geom_names:
                texture = Texture(name=name, id=id, info=dict())
                self.texture_collection.add(texture)
        # import ipdb; ipdb.set_trace()


    def view(self):
        if not self.is_Offscreen:
            self.viewer.render(); return 0
        img_ran  = self.image.random_nonfix
        img_can  = self.image.canonical
        img_diff = img_ran - img_can
        cv2.imshow(self.cv2_window_name, np.concatenate([img_ran, img_can, img_diff], axis=1))
        cv2.waitKey(50)



    def _render_flip_convert_color(self, camera_name):
        img = self.sim.render(width=self.width_capture, height=self.height_capture, camera_name=camera_name, depth=False)
        img = img[::-1]     # flip
        img = img[:,:,::-1] # reverse_channel
        return ImageObject(img)


    def set_light_on(self, use_light_index_list):
        self.model.light_active[:] = 0
        for i in use_light_index_list:
            self.model.light_active[i] = 1


    def _set_texture_rand_all_with_return_info(self):
        self.texture = {}
        max_id       = max(self.texture_collection.get_id())
        #  系列ごとのランダム化に使用するtextureを作成
        for id in range(max_id+1):
            self.texture[str(id)] = self.texture_modder.get_rand_texture()
        # texture_collectionに作成したtextureの情報を反映させる
        for id in range(max_id+1):
            self.texture_collection.assign_info_with_id(id=id, info=self.texture[str(id)])


    def _set_texture_rand_task_irrelevant_with_return_info(self):
        self.texture = {}
        max_id       = max(self.texture_collection.get_id())
        #  系列ごとのランダム化に使用するtextureを作成
        for id in range(max_id+1):
            if not id == max_id:
                self.texture[str(id)] = self.texture_modder.get_rand_texture()
        # texture_collectionに作成したtextureの情報を反映させる
        for id in range(max_id+1):
            if not id == max_id:
                self.texture_collection.assign_info_with_id(id=id, info=self.texture[str(id)])



    def _set_texture_static_all(self):
        for texture in self.texture_collection.texture:
            self.texture_modder.my_set_texture(texture.name, texture.info, is_noise_randomize=self.is_noise_randomize_per_step)


    def _set_texture_rand_all(self):
        for name in self.visible_geom:
            self.texture_modder.rand_all(name)


    def _randomize_texture(self):
        if self.randomize_texture_mode == "loaded_static":
           self._set_texture_static_all()

        elif (self.randomize_texture_mode == "per_reset") or (self.randomize_texture_mode == "static"):
            if self.is_texture_randomized is False:
                self._set_texture_rand_all_with_return_info()
                self.is_texture_randomized = True
            self._set_texture_static_all()

        elif self.randomize_texture_mode == "per_step":
            '''
                ・self.texture_collection の状態に注意
                ・ユニークな self.texture_collection を applyすることでtextureを変更している
            '''
            if self.is_texture_randomized is False:
                self._set_texture_rand_all_with_return_info()
                self.is_texture_randomized = True
            self._set_texture_rand_task_irrelevant_with_return_info()
            self._set_texture_static_all()


    def task_relevant_randomize_texture(self):
        max_id = max(self.texture_collection.get_id()) # 多分バルブ環境でしか動かない（ハードコーディング部分）
        for texture in self.texture_collection.get_textures_from_id(id=max_id):
            # import ipdb; ipdb.set_trace()
            print(texture.info)
            import ipdb; ipdb.set_trace()
            self.texture_modder.my_set_texture(texture.name, texture.info, is_noise_randomize=False)
        import ipdb; ipdb.set_trace()


    def set_rgb(self):
        for texture in self.texture_collection.texture:
            self.texture_modder.set_rgb(texture.name, texture.info["rgb"])



    def set_light_castshadow(self, shadowsize):
        self.model.vis.quality.shadowsize = shadowsize
        is_castshadow = 0 if shadowsize==0 else 1
        for name in self.model.light_names:
            # import ipdb; ipdb.set_trace()
            self.light_modder.set_castshadow(name, is_castshadow)


    def render_env(self, canonical_rgb_dict):
        if not self.is_Offscreen: return None
        self.image = Image(
            canonical     = self._render_canonical(canonical_rgb_dict),
            random_nonfix = self._render_randomized(),
            mode          = "step"
        )
        return self.image


    def __canonicalize_texture(self, canonical_rgb_dict):
        for name, rgb in canonical_rgb_dict.items():
            self.texture_modder.set_rgb(name, rgb)


    def _render_canonical(self, canonical_rgb_dict):
        shadowsize = 0
        self.__canonicalize_texture(canonical_rgb_dict)
        # self.task_relevant_randomize_texture() # あってもなくても動く
        self.sim.model.light_ambient[:] = 0    # 試す
        self.set_light_castshadow(shadowsize=shadowsize)
        self.set_light_on(self.light_index_list)
        return self._render_flip_convert_color("canonical").channel_last


    def _render_randomized(self):
        shadowsize = 0
        self._randomize_texture()
        self.sim.model.light_ambient[:] = 0 # 試す
        self.set_light_castshadow(shadowsize=shadowsize)
        self.set_light_on(self.light_index_list)
        return self._render_flip_convert_color("random_nonfix").channel_last



    def __set_camera_position(self, camera_parameter: dict):
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



    def __set_light_position(self, light_position: dict):
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
        end_effector_position = self.forward_kinematics.calc(robot_position).squeeze()
        task_space_positioin  = self.task_space.end2task(end_effector_position).squeeze()
        # force                 = self.get_force()
        state = DClawState(
            robot_position        = robot_position,
            object_position       = env_state.qpos[-1:],
            robot_velocity        = env_state.qvel[:9],
            object_velocity       = env_state.qvel[-1:],
            # force                 = force,
            end_effector_position = end_effector_position,
            task_space_positioin  = task_space_positioin,
        )
        return state


    def __set_dynamics_parameter(self, randparams_dict: dict) -> None:
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


    def reset_env(self, set_state, env_state):
        self._reset_texture_randomization_state()
        self._create_mujoco_related_instance()
        self.sim.reset()
        self._set_environment_parameters()
        set_state(env_state)
        self.sim.step()


    def _reset_texture_randomization_state(self):
        if self.randomize_texture_mode != "static":
            self.is_texture_randomized = False


    def _create_mujoco_related_instance(self):
        if self.sim is not None: return 0
        self.sim             = mujoco_py.MjSim(self.model) ; print(" init --> MjSim")
        self.texture_modder  = TextureModder(self.sim)     ; print(" init --> TextureModder")
        self.camera_modder   = CameraModder(self.sim)      ; print(" init --> CameraModder")
        self.light_modder    = LightModder(self.sim)       ; print(" init --> LightModder")
        self.__createTexutureCollection()                  ; print(" init --> TexutureCollection()")
        self.__create_viewer()                             ; print(" init --> MjViewer")


    def _set_environment_parameters(self):
        self.__set_jnt_range()
        self.__set_ctrl_range()
        self.__set_dynamics_parameter(self.dynamics)
        self.__set_camera_position(self.camera)
        self.__set_light_position(self.light)
        self.__set_target_visible(self.is_target_visible)


    def __create_viewer(self):
        if not self.is_Offscreen:
            self.viewer = mujoco_py.MjViewer(self.sim); time.sleep(1); return 0
        self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, 0); time.sleep(1)
        self.cv2_window_name = 'viewer'
        cv2.namedWindow(self.cv2_window_name, cv2.WINDOW_NORMAL)



    def set_target_position(self, target_position):
        '''
            ・バルブの目標状態の値をセット
            ・target_position: 1次元の数値
            ・renderするときに _target_position が None でなければ描画されます
        '''
        target_position       = float(target_position)
        self._target_position = target_position
        self.sim.model.body_quat[self._target_bid] = euler2quat(0, 0, float(self._target_position))


    def __set_target_visible(self, is_visible):
        if is_visible:
            if self.env_name == "blue": self.sim.model.site_rgba[self._target_sid] = [1.,  0.92156863, 0.23137255, 1]
            else                      : self.sim.model.site_rgba[self._target_sid] = [0, 0, 1, 1]
        else:
            self.sim.model.site_rgba[self._target_sid] = [0, 0, 1, 0]


    def set_ctrl_joint(self, ctrl):
        assert ctrl.shape == (9,), '[expected: {0}, input: {1}]'.format((9,), ctrl.shape)
        self.sim.data.ctrl[:9] = ctrl


    def __set_jnt_range(self):
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


    def __set_ctrl_range(self):
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

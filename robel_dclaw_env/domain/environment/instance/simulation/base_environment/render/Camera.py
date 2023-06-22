import numpy as np
from mujoco_py.modder import CameraModder
from transforms3d.euler import euler2quat


class Camera:
    def __init__(self,
            sim,
            camera_name_list,
            x_coordinate,
            y_coordinate,
            z_distance,
            orientation,
        ):
        self.camera_modder    = CameraModder(sim)
        self.camera_name_list = camera_name_list
        self.x_coordinate     = x_coordinate
        self.y_coordinate     = y_coordinate
        self.z_distance       = z_distance
        self.orientation      = orientation
        self.set_camera_posture()


    def set_camera_posture(self):
        self.__set_xyz_position()
        self.__set_quaternion()
        self.camera_modder.sim.step()


    def __set_xyz_position(self):
        for camera_name in self.camera_name_list:
            self.camera_modder.set_pos(
                name  = camera_name,
                value = [
                    self.x_coordinate,
                    self.y_coordinate,
                    self.z_distance,
                ],
            )


    def __set_quaternion(self):
        for camera_name in self.camera_name_list:
            self.camera_modder.set_quat(
                name  = camera_name,
                value = euler2quat(np.deg2rad(self.orientation), 0.0, np.pi/2)
            )


    # def get_camera_parameter(self, isDict: bool = False):
    #     (x, y, z)  = self.camera_modder.get_pos(name="cam_canonical_pos_nonfix")
    #     quat       = self.camera_modder.get_quat(name="cam_canonical_pos_nonfix")
    #     params     = {
    #             "x_coordinate": x,
    #             "y_coordinate": y,
    #             "z_distance"  : z,
    #             "orientation" : quat2euler(quat)[0]
    #     }
    #     if isDict is False:
    #         params = dictOps.dict2numpyarray(params)
    #     return params

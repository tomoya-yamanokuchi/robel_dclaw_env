import numpy as np
# from mujoco_py.modder import LightModder


class Light:
    def __init__(self, sim, light1: int, light2: int):

        self.model        = sim.model
        self.light1       = light1
        self.light2       = light2

    def set_light_on(self):
        self.set_light_off()
        self.model.light_active[self.light1] = 1
        self.model.light_active[self.light2] = 1


    def set_light_off(self):
        self.model.light_active[:] = 0


    def set_light_ambient(self, value=0):
        self.model.light_ambient[:] = value


    def set_light_castshadow(self, shadowsize: int):
        self.model.vis.quality.shadowsize = shadowsize
        is_castshadow = 0 if shadowsize==0 else 1
        self.model.light_castshadow[:]           = 0
        self.model.light_castshadow[self.light1] = is_castshadow
        self.model.light_castshadow[self.light2] = is_castshadow


    def get_light_parameter(self, isDict: bool = False):
        if not isDict: return np.array([self.light1, self.light2])
        return {
            "light1" : self.light1,
            "light2" : self.light2,
        }

import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from ..base_environment.dynamics_parameter.AbstractDynamicsParameter import AbstractDynamicsParameter



class ValveDyanmicsParameter(AbstractDynamicsParameter):
    def __init__(self, sim):
        self.sim = sim


    def set(self, randparams_dict: dict) -> None:
        set_dynamics_parameter_function = {
            "kp_valve"           :  self._set_valve_actuator_gain_position,
            "kv_valve"           :  self._set_valve_actuator_gain_velocity,
            "damping_valve"      :  self._set_valve_damping,
            "frictionloss_valve" :  self._set_valve_frictionloss,
        }
        for key, value in randparams_dict.items():
            set_dynamics_parameter_function[key](value)


    def _set_valve_actuator_gain_position(self, kp):
        self.sim.model.actuator_gainprm[-2, 0] =  kp
        self.sim.model.actuator_biasprm[-2, 1] = -kp


    def _set_valve_actuator_gain_velocity(self, kv):
        self.sim.model.actuator_gainprm[-1, 0] =  kv
        self.sim.model.actuator_biasprm[-1, 2] = -kv


    def _set_valve_damping(self, value):
        self.sim.model.dof_damping[-1] = value


    def _set_valve_frictionloss(self, value):
        self.sim.model.dof_frictionloss[-1] = value

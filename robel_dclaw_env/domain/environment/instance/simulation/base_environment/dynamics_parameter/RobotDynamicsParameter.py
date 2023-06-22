from .AbstractDynamicsParameter import AbstractDynamicsParameter


class RobotDynamicsParameter(AbstractDynamicsParameter):
    def __init__(self, sim):
        self.sim = sim


    def set(self, randparams_dict: dict) -> None:
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

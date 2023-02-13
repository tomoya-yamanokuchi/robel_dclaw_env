from abc import ABCMeta, abstractmethod

'''
    ・環境の抽象クラスです
    ・具象クラスはこの抽象クラスを継承し，抽象クラス内で定義されているメソッドを実装する必要があります
    ・OpenAI gym などの特定の環境を再現しているわけではないので，一般的に公開されているもととは
      少し異なっているかもしれません
'''

class AbstractEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config):
        pass
        """
        - load_model
        -
        """

    @abstractmethod
    def reset(self):
        pass
        """
        - reset
            - _create_mujoco_related_instance
                - _set_geom_names_randomize_target
            - set_jnt_range
            - set_ctrl_range
            - set_dynamics_parameter
                - set_claw_actuator_gain_position
                - set_claw_damping
                - set_claw_frictionloss
                - set_valve_actuator_gain_position
                - set_valve_actuator_gain_velocity
                - set_valve_damping
                - set_valve_frictionloss
            - set_camera_position
            - set_light_position
            - set_target_visible
            - _create_qpos_qvel_from_InitialState
            - set_state
            - render
        # _render
        """

    @abstractmethod
    def set_ctrl_joint(self):
        pass
        """
        - set_ctrl
        """

    @abstractmethod
    def set_ctrl_task_diff(self):
        pass


    @abstractmethod
    def get_state(self):
        pass
        """
        - get_state
            - get_force
        """


    @abstractmethod
    def step(self):
        pass
        """
        - step
        - step_with_inplicit_step
        """


    @abstractmethod
    def render(self):
        pass
        """
        - render
            - render_with_viewer
                - canonicalize_texture
                - randomize_texture
                    - _set_texture_rand_all
                    - _set_texture_rand_all_with_return_info
                    - _set_texture_static_all
                - set_light_castshadow
                - set_light_on
                - _render_and_convert_color
                    - _flip
                    - _reverse_channel
        """


    @abstractmethod
    def set_target_position(self):
        pass
        """
        set_target_position
        """


    #
    """
    debug function
    - check_camera_pos
    - set_camera_position_with_all_euler
    """


    """
    data collection for model traing
    - get_camera_parameter
    - get_dynamics_parameter
    - get_light_parameter
    """

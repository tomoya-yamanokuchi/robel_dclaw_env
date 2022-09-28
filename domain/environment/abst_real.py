from abc import ABCMeta, abstractmethod




class AbstractEnvironment(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config):
        pass
        """
        - set_node_name
        - initialize_ros_node
        - @Subscriber
        - @Publisher
        - @CameraROSHandler
            - set_node_name(node_name)
            - initialize_ros_node()
            - __Subscriber = CameraSubscriber()
            - __Publisher  = CameraPublisher(sleep_time_sec=publisher_sleep_time_sec)
            - wait_ros_connection_establishment()
            - print_initialized_message()
        - wait_ros_connection_establishment
        - print_initialized_message
        """


    @abstractmethod
    def reset(self):
        pass
        """
        - reset_env
            - publish_initialize_ctrl
        """


    @abstractmethod
    def set_ctrl(self):
        pass
        """
        - publish_joint_ctrl
        - publish_valve_ctrl
        """


    @abstractmethod
    def get_state(self):
        pass
        """
        - get_joint_currents
        - get_joint_positions
        - get_joint_velocities
        - get_valve_moving
        - get_valve_position
        """


    @abstractmethod
    def step(self):
        pass
        """
        - self.ros_handler.publish_joint_ctrl(self.ctrl)
            - self.pub_joint_ctrl.publish(self.msg_joint_ctrl)
            - rospy.sleep(self.sleep_time_sec)
        - self._step_task_space() -> env自体には含まれるべきでない
        """

        """
        - step は 解釈性の観点から rospy.sleep であるべき（今のpublishとsleepが一体になっているのは変な感じがする）
        -
        """


    @abstractmethod
    def set_target_position(self):
        pass
        """
        set_target_position
        """
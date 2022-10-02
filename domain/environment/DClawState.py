from dataclasses import dataclass
import numpy as np

'''
・Dclaw環境に状態を与える時に使用するクラスです
・与えるべき状態のルールが記述されています
'''

@dataclass(frozen=True)
class DClawState:
    robot_position       : np.ndarray
    object_position      : np.ndarray
    robot_velocity       : np.ndarray
    object_velocity      : np.ndarray
    # force                : np.ndarray
    end_effector_position: np.ndarray
    task_space_positioin : np.ndarray

    def __post_init__(self):
        assert type(self.robot_position) == np.ndarray
        assert self.robot_position.shape == (9,)

        val_type = type(self.object_position)
        if   val_type == np.ndarray :   assert self.object_position.shape == (1,)
        elif val_type == float      :   pass
        else                        :   raise NotImplementedError()

        assert type(self.robot_velocity) == np.ndarray
        assert self.robot_velocity.shape == (9,)

        val_type = type(self.object_velocity)
        if   val_type == np.ndarray :   assert self.object_velocity.shape == (1,)
        elif val_type == float      :   pass
        else                        :   raise NotImplementedError()

        # assert type(self.force) == np.ndarray
        # assert self.force.shape == (9,)

        # assert type(self.end_effector_position) == np.ndarray
        # assert self.end_effector_position.shape == (9,)


if __name__ == '__main__':
    import numpy as np

    state = DClawState(
        robot_position        = np.zeros(9),
        object_position       = np.zeros(1),
        robot_velocity        = np.zeros(9),
        object_velocity       = np.zeros(1),
        force                 = np.random.randn(9),
        end_effector_position = np.zeros(9),
    )
    print(state.robot_position)
    print(state.object_position)
    print(state.force)

    state = DClawState(
        robot_position        = np.zeros(9),
        object_position       = 0.0,
        robot_velocity        = np.zeros(9),
        object_velocity       = 0.0,
        force                 = np.random.randn(9),
        end_effector_position = np.zeros(9),
    )
    print(state.robot_position)
    print(state.object_position)
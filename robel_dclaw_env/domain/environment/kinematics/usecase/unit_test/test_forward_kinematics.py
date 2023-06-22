import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from kinematics.domain import NumpyForwardKinematics, TorchForwardKinematics
from kinematics.service import to_numpy, to_tensor
import numpy as np


class Test:
    def __init__(self):
        self.numpy_forward = NumpyForwardKinematics()
        self.torch_forward = TorchForwardKinematics()


    def test_forward_kenimatics_1claw(self, joint_position):
        out_numpy = self.numpy_forward.calc_1claw(joint_position)
        out_torch = self.torch_forward.calc_1claw(to_tensor(joint_position).cuda())
        np.testing.assert_almost_equal(
            actual  = out_numpy,
            desired = to_numpy(out_torch),
            decimal = 5
        )
        print("<< pass unit test 1claw >>")


    def test_forward_kenimatics_all_claw(self, joint_position):
        out_numpy = self.numpy_forward.calc(joint_position)
        out_torch = self.torch_forward.calc(to_tensor(joint_position).cuda())
        np.testing.assert_almost_equal(
            actual  = out_numpy,
            desired = to_numpy(out_torch),
            decimal = 5
        )
        print("<< pass unit test all claw>>")



if __name__ == '__main__':
    test = Test()

    joint_position = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, np.pi*0.5, 0.0],
            [0.0, -np.pi*0.5, 0.0],
            [0.0, 0.0, np.pi*0.5],
            [0.0, 0.0, -np.pi*0.5],
            [np.pi*0.5, 0.0, 0.0],
            # [-np.pi*0.5, 0.0, 0.0], # ダメな角
            # [numpy_forward.kinematics.theta0_lb+1.5, 0.0, 0.0], # ダメな角度
        ]
    )
    test.test_forward_kenimatics_1claw(joint_position)


    joint_position = np.array(
        [
            [0.0, -np.pi*0.5 , np.pi*0.5, 0.0, -np.pi*0.5 , np.pi*0.5, 0.0, -np.pi*0.5 , np.pi*0.5],
        ]
    )
    test.test_forward_kenimatics_all_claw(joint_position)

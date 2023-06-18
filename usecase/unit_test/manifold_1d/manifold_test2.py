import numpy as np
import sys; import pathlib; p = pathlib.Path("./"); sys.path.append(str(p.cwd())); sys.path.insert(0, './robel_dclaw_env')
from robel_dclaw_env.domain.environment.task_space.manifold_1d.Manifold1D import Manifold1D as TorchTaskSpace
from robel_dclaw_env.domain.environment.task_space.manifold_1d_numpy.Manifold1D import Manifold1D as NumpyTorchTaskSpace
from robel_dclaw_env.domain.environment.task_space.manifold_1d.TaskSpacePositionValue_1D_Manifold import TaskSpacePositionValue_1D_Manifold as TaskSpaceValueObject
from torch_numpy_converter import to_numpy, to_tensor_double, NTD

class Test:
    def __init__(self):
        self.numpy_task_space = NumpyTorchTaskSpace()
        self.torch_task_space = TorchTaskSpace()


    def run(self, task_space_value):
        numpy_smd = self.numpy_task_space.task2signed_distance_matrix(TaskSpaceValueObject(NTD(task_space_value)))
        torch_smd = self.torch_task_space.task2signed_distance_matrix(TaskSpaceValueObject(NTD(to_tensor_double(task_space_value))))

        np.testing.assert_almost_equal(
            actual  = numpy_smd[0],
            desired = to_numpy(torch_smd[0]),
            decimal = 16
        )
        print("<< pass unit test 1claw >>")


if __name__ == '__main__':
    test = Test()


    t = np.linspace(start=0.0, stop=1.0, num=400)
    t = t.reshape(-1, 1)
    t = np.tile(t, (1, 3))

    test.run(t)



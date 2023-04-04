import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from custom_service import concat, NTD


class ValveReference:
    def __init__(self, planning_horizon):
        self.planning_horizon = planning_horizon
        self.task_horizon     = 20
        self.d_theta          = (2*np.pi) / self.task_horizon


    def get_timesteps(self, current_step):
        timesteps = np.linspace(
            start = current_step + 1,
            stop  = current_step + self.planning_horizon,
            num   = self.planning_horizon,
            dtype = int,
        )
        assert np.diff(timesteps).sum() == (self.planning_horizon - 1)
        return timesteps


    def get_as_radian(self, current_step: int):
        theta     = self.d_theta * current_step
        reference = None
        for h in range(self.planning_horizon):
            theta    += self.d_theta
            reference = concat(reference, np.array([theta]), axis=0)
        reference = reference.reshape(1, -1, 1)
        return reference


    def get_as_polar_coordinates(self, current_step: int):
        radian    = self.get_as_radian(current_step)
        reference = np.stack((np.cos(radian), np.sin(radian)), axis=-1)
        return reference


if __name__ == '__main__':
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    import matplotlib.pyplot as plt
    from matplotlib import ticker, cm


    reference = ValveReference(
        planning_horizon = 25,
    )

    ref       = reference.get_as_radian(current_step=0)
    timesteps = reference.get_timesteps(current_step=0)
    plt.plot(timesteps, ref.squeeze(), "-x")
    plt.plot(timesteps, np.zeros_like(timesteps) + np.pi*2, "-o")
    plt.xlim(0, 30)
    plt.ylim(0, None)
    plt.grid()
    plt.show()

    ref       = reference.get_as_polar_coordinates(current_step=0)
    timesteps = reference.get_timesteps(current_step=0)
    plt.plot(timesteps, ref.squeeze(), "-x")
    # plt.plot(timesteps, np.zeros_like(timesteps) + np.pi*2, "-o")
    plt.xlim(0, 30)
    plt.ylim(-1, 1)
    plt.grid()
    plt.show()

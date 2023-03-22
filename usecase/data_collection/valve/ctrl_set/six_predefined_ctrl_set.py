import numpy as np


class SixPredefinedCtrlSet:
    # def __init__(self):
        # self


    def get(self, step):
        const = 0.05
        ctrl_task_diff = np.stack(
            (
                np.stack((np.zeros(step) + const, np.zeros(step),           np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step) - const, np.zeros(step),           np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step) + const,   np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step) - const,   np.zeros(step))         , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step),           np.zeros(step) + const) , axis=-1),
                np.stack((np.zeros(step),         np.zeros(step),           np.zeros(step) - const) , axis=-1),
            ), axis=0
        )

        ctrl = np.cumsum(ctrl_task_diff, axis=1)

        return ctrl




import os
import time
import numpy as np
import multiprocessing
from .UnitProcess import UnitProcess

class UnitProcessCollection:
    def __init__(self):
        self.collection = []


    def add(self, unit_process: UnitProcess):
        assert isinstance(unit_process, UnitProcess)
        self.collection.append(unit_process)


    def size(self):
        return len(self.collection)


    def num_alive(self):
        boolean_alive_list = [process.is_alive() for process in self.collection]
        return np.sum(np.array(boolean_alive_list)*1)


    def join(self):
        for process in self.collection:
            process.join()
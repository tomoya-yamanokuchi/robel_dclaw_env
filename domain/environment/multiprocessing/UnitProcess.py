import os
import time
import multiprocessing
from .EnvironmentConstantSetting import EnvironmentConstantSetting



class UnitProcess:
    def __init__(self, id: int, function, constant_setting: EnvironmentConstantSetting):
        assert type(id) == int
        assert isinstance(constant_setting, EnvironmentConstantSetting)
        self.id           = id
        self.queue_input  = multiprocessing.Queue(1)
        self.queue_result = multiprocessing.Queue(1)
        self.process      = multiprocessing.Process(
            target = function,
            args   = (constant_setting,),
        )


    def start(self, ctrl):
        print("******* start ", ctrl.shape)
        self.process.daemon = True
        self.process.start()
        print("qqqqqqqqqqqqqqqqqqqqqqqq")
        # self.queue_input.put((self.id, ctrl))
        # self.queue_input.join()
        print("++++++++++++++++++")


    def is_alive(self):
        return self.process.is_alive()


    def join(self):
        self.process.join()


    def stop(self):
        self.process.terminate()
        self.process.close()



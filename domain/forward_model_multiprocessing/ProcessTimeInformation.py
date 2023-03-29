import time



class ProcessTimeInformation:
    def __init__(self, verbose=True):
        self.time_init = None
        self.time_end  = None
        self.verbose   = verbose


    def time_start(self):
        self.time_init = time.time()


    def time_stop(self):
        self.time_end = time.time()
        proc_time     = self.time_end - self.time_init
        if self.verbose: self._print_process_time(proc_time)
        return proc_time


    def _print_process_time(self, proc_time):
        print("-------------------------------")
        print("   procces time : {: .3f} [sec]".format(proc_time))
        print("-------------------------------")


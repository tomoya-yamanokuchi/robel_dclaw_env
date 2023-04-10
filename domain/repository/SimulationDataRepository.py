import os
import datetime
import shelve
import pathlib
import numpy as np
import glob
from pprint import pprint
from natsort import natsorted
from domain.forward_model_multiprocessing.ForkedPdb import ForkedPdb
from .CollectionAggregation import CollectionAggregation


class SimulationDataRepository:
    def __init__(self,
            dataset_dir :str  = "./dataset",
            dataset_name:str  = None,
            read_only   :bool = False,
            verbose     :bool = False,
        ):
        self.dataset_name     = dataset_name
        self.read_only        = read_only
        self.dataset_dir      = dataset_dir
        self.verbose          = verbose
        self.dataset_save_dir = self.__create_dataset_dir()


    def __create_parent_dir(self):
        self.p = pathlib.Path(self.dataset_dir)
        self.p.mkdir(parents=True, exist_ok=True)


    def __create_dataset_dir(self):
        self.__create_parent_dir()
        if self.dataset_name is None:
            date              = datetime.datetime.now()
            self.dataset_name = "dataset_{}{}{}{}{}{}".format(date.year, date.month, date.day, date.hour, date.minute, date.second)
        dataset_save_dir = str(self.p.resolve()) + "/" + self.dataset_name
        os.makedirs(dataset_save_dir, exist_ok=True)
        return dataset_save_dir


    def open(self, filename:str = "menu_data"):
        '''
        Value :   Meaning
        -----------------
        'r'   : Open existing database for reading only (default)
        'w'   : Open existing database for reading and writing
        'c'   : Open database for reading and writing, creating it if it doesn’t exist
        'n'   : Always create a new, empty database, open for reading and writing
        '''
        flag            = 'r' if self.read_only else 'c'
        if ".db" in filename: filename = filename.split(".")[0]
        full_path       = self.dataset_save_dir + '/' + filename
        self.repository = shelve.open(full_path, flag=flag) # read only
        if self.verbose: print("shelve.open (flag={}) --> {}".format(flag, full_path))


    def close(self):
        self.repository.close()


    def get_filenames(self):
        pathlib_object = pathlib.Path(self.dataset_save_dir)
        path_list      = glob.glob(os.path.join(str(pathlib_object), "*"))
        path_list      = natsorted(path_list)
        # import ipdb; ipdb.set_trace()
        filenames      = [path.split("/")[-1] for path in path_list]
        return filenames


    def assign(self, dataclass_list, name):
        agg  = CollectionAggregation()
        data = agg.aggregate(dataclass_list)
        self.repository[name] = data
        # ForkedPdb().set_trace()

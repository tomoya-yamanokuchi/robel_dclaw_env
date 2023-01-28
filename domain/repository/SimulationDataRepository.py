import os
import psutil
import datetime
import shelve
import pathlib
import numpy as np
from natsort import natsorted
from .dataclass_concatenate import dataclass_concatenate


class SimulationDataRepository:
    def __init__(self, dataset_dir:str="./dataset", dataset_name:str=None, read_only:bool=False):
        self.dataset_name     = dataset_name
        self.read_only        = read_only
        self.dataset_dir      = dataset_dir
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


    def open(self, filename="menu_data"):
        '''
        Value :   Meaning
        -----------------
        'r'   : Open existing database for reading only (default)
        'w'   : Open existing database for reading and writing
        'c'   : Open database for reading and writing, creating it if it doesn’t exist
        'n'   : Always create a new, empty database, open for reading and writing
        '''
        flag            = 'r' if self.read_only else 'c'
        full_path       = self.dataset_save_dir + '/' + filename
        time_now        = datetime.datetime.now()
        self.repository = shelve.open(full_path, flag=flag)
        # print("[{}] shelve.open (flag={}) --> {}".format(time_now, flag, full_path))

        # -------------メモリ使用量を取得 ----------------
        # mem = psutil.virtual_memory()
        # np.save(self.dataset_save_dir + '/' + filename + "_mem", np.array(mem.percent))


    def close(self):
        self.repository.close()


    def assign(self, key: str, dataclass_list: list, cls: object):
        assert type(key) == str
        assert type(dataclass_list) == list
        sequencial_dataclass = dataclass_concatenate(dataclass_list, cls) # リストオブジェクトを１つのオブジェクトにまとめる
        self.repository[key] = vars(sequencial_dataclass)                 # shelveの保存するものとしてクラスのフィールド値を辞書として取得

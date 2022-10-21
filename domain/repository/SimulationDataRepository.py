import os
import psutil
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
        if self.dataset_name is not None:
            dataset_save_dir = str(self.p.resolve()) + "/" + self.dataset_name
        else:
            # 存在するディレクトリ名の一覧を取得
            existing_dirs = os.listdir(str(self.p.resolve()))
            existing_dirs = natsorted(existing_dirs)
            # 作成するディレクトリ名を決定
            if existing_dirs == []:
                dataset_name = "dataset_0"
            else:
                splitted_name = existing_dirs[-1].split('_') #ハードコーディング部分，命名規則に依存する
                name          = splitted_name[0]
                identifier    = splitted_name[-1]
                # import ipdb; ipdb.set_trace()
                print("identifier --> ", identifier)
                new_identifier   = str(int(identifier) + 1)
                dataset_name     = '_'.join([name, "autoname", new_identifier, ])
            dataset_save_dir = str(self.p.resolve()) + "/" + dataset_name

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
        # print("=======================")
        # print(" open shelv as: " + flag)
        # print("=======================")

        print(self.dataset_save_dir + '/' + filename)

        self.repository = shelve.open(self.dataset_save_dir + '/' + filename, flag=flag)

        # # メモリ使用量を取得 ----------------
        # mem = psutil.virtual_memory()
        # np.save(self.dataset_save_dir + '/' + filename + "_mem", np.array(mem.percent))


    def close(self):
        self.repository.close()


    def assign(self, key, dataclass_list, cls):
        self.repository[key] = dataclass_concatenate(dataclass_list, cls)

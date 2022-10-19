import os
import shelve
import pathlib
from natsort import natsorted
from .dataclass_concatenate import dataclass_concatenate


class SimulationDataRepository:
    def __init__(self, dataset_dir="./dataset", dataset_name=None):
        self.dataset_name     = dataset_name
        self.read_only        = True if dataset_name is not None else False
        self.dataset_dir      = dataset_dir
        self.dataset_save_dir = self.__create_dataset_dir(dataset_dir)


    def __create_dataset_dir(self, dataset_dir):
        # 存在するディレクトリ名の一覧を取得
        p             = pathlib.Path(dataset_dir)
        p.mkdir(parents=True, exist_ok=True)
        existing_dirs = os.listdir(str(p.resolve()))
        existing_dirs = natsorted(existing_dirs)
        # 作成するディレクトリ名を決定
        if self.read_only:
            dataset_name = self.dataset_name
        else:
            if existing_dirs == []:
                dataset_name = "dataset_0"
            else:
                name, identifier = existing_dirs[-1].split('_')
                new_identifier   = str(int(identifier) + 1)
                dataset_name     = '_'.join([name, new_identifier])
        dataset_save_dir = str(p.resolve()) + "/" + dataset_name
        # 新たなディレクトリを作成
        if not self.read_only:
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
        self.repository = shelve.open(self.dataset_save_dir + '/' + filename, flag=flag)


    def close(self):
        self.repository.close()


    def assign(self, key, dataclass_list, cls):
        self.repository[key] = dataclass_concatenate(dataclass_list, cls)

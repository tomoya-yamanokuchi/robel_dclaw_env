import numpy as np
from custom_service import concat



class CollectionAggregation:
    def __init__(self, dataclass_list):
        self.dataclass_list = dataclass_list


    def aggregate(self):
        dataclass_fields = list(self.dataclass_list[0].__dict__.keys())
        self._remove_unnecessary_key(dataclass_fields)
        aggregate_data = self._concatenate(dataclass_fields)
        return aggregate_data


    def _remove_unnecessary_key(self, dataclass_fields):
        if 'mode' in dataclass_fields:
            dataclass_fields.remove('mode') # modeを結合から外す+


    def _concatenate(self, dataclass_fields):
        field_dict = self.__initialize_field_dict(dataclass_fields)
        field_dict = self.__concat_data(field_dict, dataclass_fields)
        return field_dict


    def __initialize_field_dict(self, dataclass_fields):
        #list内の各dataclassのフィールド値を保存するための辞書を初期化
        field_dict = dict()
        for field in dataclass_fields:
            field_dict[field] = None
        return field_dict


    def __concat_data(self, field_dict, dataclass_fields):
        for dataclass in self.dataclass_list:
            for field in dataclass_fields:
                # print("field = ", field)
                val = dataclass.__dict__[field]
                if val is not None:
                    field_dict[field] = concat(field_dict[field], val[np.newaxis], axis=0)
        return field_dict


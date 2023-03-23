import numpy as np


def dataclass_concatenate(dataclass_list: list, cls):
    # cls内のフィールド値を取得
    # dataclass_fields = list(cls.__dataclass_fields__.keys())
    dataclass_fields = list(cls.__dict__.keys())

    dataclass_fields.remove('mode') # modeを結合から外す

    # list内の各dataclassのフィールド値を保存するための辞書を初期化
    field_dict = dict()
    for field in dataclass_fields:
        field_dict[field] = []

    # feild値を全てのdataclassから取り出しリストに格納
    for dataclass in dataclass_list:
        assert isinstance(dataclass, cls)
        for field in dataclass_fields:
            val = dataclass.__dict__[field]
            field_dict[field].append(val)

    # リストをstaskして系列データに変換
    for field in dataclass_fields:
        field_dict[field] = np.stack(field_dict[field], axis=0)

    # 系列化した各フィールドを用いて新たにdataclassを作成
    return cls(**field_dict, mode="sequence")


import numpy as np
from robel_dclaw_env.custom_service import concat

import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from robel_dclaw_env.domain.forward_model_multiprocessing.ForkedPdb import ForkedPdb


class CollectionAggregation:
    def _initialize_aggregation_dict(self, keys):
        data = {}
        for key in keys:
            data[key] = []
        return data


    def aggregate(self, dataclass_list: list):
        keys = dataclass_list[0].collection.keys()
        data = self._initialize_aggregation_dict(keys)
        # -----
        for t in range(len(dataclass_list)):
            x = dataclass_list[t]
            for key in keys:
                print("key = ", key)
                val = x.collection[key].value
                val = val.squeeze() if type(val) == np.ndarray else val
                data[key].append(val)
        # -----
        for key in keys:
            data[key] = np.stack(data[key])
        # -----
        # ForkedPdb().set_trace()
        return data

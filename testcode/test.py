import os
import numpy as np
from natsort import natsorted
import pprint


dir = "/home/tomoya-y/workspace/robel-dclaw-env/dataset/dataset_20221020122849"

files = os.listdir(dir)
files = [f for f in files if "_mem" in f]
files = natsorted(files)

pprint.pprint(files)

mem = np.zeros(len(files))

for i in range(len(files)):
    m = np.load(dir + "/" + files[i])
    import ipdb; ipdb.set_trace()

# for i in range'()
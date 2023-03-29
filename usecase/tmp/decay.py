
import numpy as np


horizon    = 10
gamma      = 0.95
time_decay = np.array([gamma**t for t in range(horizon)])


print(time_decay)

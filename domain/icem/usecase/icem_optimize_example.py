import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.icem.iCEM_MPC import iCEM_MPC
from domain.icem.cost_function_example import cos_sin_similarity_cost_function
from domain.icem.forward_model_example import forward_model_with_fixed_initial_point


horizon = 30

cost_fn       = lambda y: cos_sin_similarity_cost_function(y, horizon=horizon)
forward_model = forward_model_with_fixed_initial_point


icem = iCEM_MPC(
    forward_model          = forward_model,
    num_sample             = 100,
    num_elite              = 10,
    num_cem_iter           = 5,
    planning_horizon       = horizon,
    dim_action             = 2,
    colored_noise_exponent = 2.0,
    fraction_rate_elite    = 0.3,
    decay_sample           = 1.25,
    lower_bound            = -1.0, # -0.1,
    upper_bound            = 1.0, # 0.1,
    alpha                  = 0.1,
    verbose                = True,
)


state = (-0.2, 0.6)

icem.reset()


for t in range(10):
    cost = icem.optimize(
        state         = state,
        cost_function = cost_fn,
    )

cost_min  = icem.cost_history.get_cost_min()
cost_max  = icem.cost_history.get_cost_min()
cost_mean = icem.cost_history.get_cost_min()

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib import ticker, cm

plt.plot(cost_min , label="min")
plt.plot(cost_max , label="max")
plt.plot(cost_mean, label="mean")
plt.legend()
plt.show()
# plt.savefig("./cost.png")

# print("cost = ", cost)

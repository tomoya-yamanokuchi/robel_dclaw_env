import numpy as np
import sys; import pathlib; p = pathlib.Path(); sys.path.append(str(p.cwd()))
from domain.icem.cost_function_example import cos_sin_similarity_cost_function
from domain.icem.forward_model_example import forward_model_with_fixed_initial_point


def sample():
    x      = np.linspace(0, 2*np.pi, 100)
    cos    = np.cos(x).reshape(-1, 1)
    sin    = np.cos(x).reshape(-1, 1)
    sample = np.concatenate((cos, sin), axis=-1)
    return sample


cost_fn       = cos_sin_similarity_cost_function
forward_model = forward_model_with_fixed_initial_point

sample_model_input = sample()
y = forward_model(sample_model_input, state=(-0.2, 0.6))

loss_sample_model_input = cost_fn(sample_model_input)
loss_model              = cost_fn(y)

print("---------------------------------------------------")
print("loss_sample_model_input = ", loss_sample_model_input)
print("             loss_model = ", loss_model)
print("---------------------------------------------------")

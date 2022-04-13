"""
Generating data with simulator_v3 (dependent skills) and fitting each skill independently with the simplebkt model.
It is expected that will fail at finding good estimates due to the wrong assumption of independent skills.
"""
import numpy as np
import torch

from torchbkt import *
from datahelper import *
from simulator import Simulator_v3


simulator_params = {
    'blocks': [5], # just use one block, i.e. [block-size]
    "n_skills": 5 # block_size
}
n_skills = simulator_params['n_skills'] # for convenience

bkt_params = {
    'l0': (0, 0, 0, 0.2, 0), # per skill
    'transition': (0.2, 0.3, 0.52, 0.07, 0.12), # per skill
    'slip': (0.2,), # per block
    'guess': (0.2,), # per block
    'max_skills': 4 # max. skills per exercise
}

sampling_params = {
    'n_students': 2000,
    'n_exercises': 60 
}

# optimization parameters
optimization_params = {
    'lr': 0.005,
    'epochs': 6,
    'batch_size': 64
}

# simulation setup
np.random.seed(2021)
torch.manual_seed(2021)
simulation_runs = 5
verbose = True


if __name__ == '__main__':
    # simulate data
    student_simulator = Simulator_v3(**simulator_params, **bkt_params)
    data = student_simulator.sample_students(**sampling_params)

    # initialize model
    model = BKTFitMultiple(data, n_skills, bkt_params['max_skills'], **optimization_params, simulator_version=3)
    df = model.fitting_multiple(verbose=True)

    # print parameter estimates
    print(df)
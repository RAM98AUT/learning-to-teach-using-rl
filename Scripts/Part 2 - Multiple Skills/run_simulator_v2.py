"""
This script uses simulator v2 to generate data and fit a BKT model. 
All skills are assumed to be independent.
"""
from simulator import Simulator_v2
from torchbkt import BKTFitMultiple


simulator_params = {
    'n_skills': 20,
    'n_blocks': 5 
}

bkt_params = {
    'l0': 0,
    'transition': (0.1,) * simulator_params['n_skills'],
    'slip': (0.05, 0.1, 0.15, 0.2, 0.25),
    'guess': (0.25, 0.2, 0.15, 0.1, 0.05)
}

sampling_params = {
    'n_students': 1000,
    'n_exercises': 100,
    'choose_skill': '_blockwise',
    'max_skills': 3
}

optimization_params = {
    'lr': 0.01,
    'epochs': 20,
    'batch_size': 8
}


if __name__=='__main__':
    student_simulator = Simulator_v2(**simulator_params, **bkt_params)
    data = student_simulator.sample_students(**sampling_params)
    model = BKTFitMultiple(data, simulator_params['n_skills'], sampling_params['max_skills'], **optimization_params)
    df = model.fitting_multiple(verbose=True)
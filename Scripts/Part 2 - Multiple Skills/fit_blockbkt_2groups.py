"""
Generating data with simulator_v3 (dependent skills) for 2 different parameter sets
and fitting a blockbkt model to the combined data.
"""

import numpy as np
import torch
import pandas as pd
import pickle

from torchbkt import *
from datahelper import *
from simulator import Simulator_v3


simulator_params = {
    'blocks': [5,3,6],
    'n_skills': 14
}
n_skills = simulator_params['n_skills'] # for convenience
n_blocks = len(simulator_params["blocks"])
blocks = simulator_params["blocks"]
max_skills=4

slow_params = {
    'l0': np.array([0.05,0.01,0.02,0.08]* 3+[0.3,0.02]),
    'transition': [0.2,0.15,0.3,0.1] *2+[0.05,0.12]*3,
    'slip': (0.1,0.2,0.1),
    'guess': (0.15,0.05,0.25),
    'blocks': simulator_params["blocks"],
    'n_skills': simulator_params["n_skills"]
}

fast_params = {
    'l0': np.array([0.05,0.01,0.02,0.08]* 3+[0.3,0.02]),
    'transition': [0.3,0.25,0.4,0.2] *2+[0.15,0.22]*3,
    'slip': (0.05,0.15,0.05),
    'guess': (0.15,0.05,0.25),
    'blocks': simulator_params["blocks"],
    'n_skills': simulator_params["n_skills"]
}

sampling_params = {
    'n_students': 1000,
    'n_exercises': 56
}

# optimization parameters
lr = 0.01
epochs = 15
batch_size = 64
step_size = 4
gamma = 0.1

delta = 1
omicron = 0

# simulation setup
np.random.seed(2021)
torch.manual_seed(2021)
verbose = True

# simulation results
slips = []
guesses = []
l0s = []
transitions = []


if __name__ == '__main__':

    # simulate data
    student_simulator = Simulator_v3(**slow_params,max_skills=max_skills)
    data_slow = student_simulator.sample_students(**sampling_params)
    
    # simulate data
    student_simulator_fast = Simulator_v3(**fast_params,min_id=sampling_params["n_students"],max_skills=max_skills)
    data_fast = student_simulator_fast.sample_students(**sampling_params)
    data_complete = pd.concat([data_slow, data_fast])
    
    for i in range(n_blocks):
        #Filter data
        data = data_complete.loc[data_complete['block_id'] == i]
        
        # initialize dataset
        C = data['correct'].values
        columns = ["skill_name" + str(j) for j in range( sum(blocks[:i]), sum(blocks[:i+1]) )]
        S = data[columns].values
        user_ids = data['user_id'].values
        train_dataset = BlockBKTDataset(C, S, user_ids)
    
        # initialize model
        model = BlockBKT(blocks[i], max_skills)
    
        # initialize trainer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        trainer = BlockBKTTrainer(model, optimizer, scheduler, device, delta, omicron)
    
        # train model
        trainer.fit(train_dataset, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # extract and store estimated parameters
        with torch.no_grad():
            # extract
            l0 = torch.softmax(model.priors, dim=1).numpy()[:, 1]
            slip = torch.softmax(model.emission.emission_matrix, dim=1).numpy()[1, 0]
            guess = torch.softmax(model.emission.emission_matrix, dim=1).numpy()[0, 1]
            transition = torch.softmax(model.transition.transition_matrix, dim=1).numpy()[:, 1]
            # store
            l0s += l0.tolist()
            slips.append(slip)
            guesses.append(guess)
            transitions +=transition.tolist()

#Saving parameters
fitted_params = {
    'l0': l0s,
    'transition': transitions,
    'slip':slips,
    'guess': guesses,
    'blocks': blocks,
    'n_skills': n_skills,
    'blocks': simulator_params["blocks"],
    'n_skills': simulator_params["n_skills"]
}

slow_params_file = 'output/torchbkt/block_params_simulated_slow.pkl'
fast_params_file = 'output/torchbkt/block_params_simulated_fast.pkl'
fitted_params_file = 'output/torchbkt/block_params_simulated_fitted.pkl'

with open(fitted_params_file, 'wb') as out:
     pickle.dump(fitted_params, out)

with open(slow_params_file, 'wb') as out:
     pickle.dump(slow_params, out)

with open(fast_params_file, 'wb') as out:
     pickle.dump(fast_params, out)



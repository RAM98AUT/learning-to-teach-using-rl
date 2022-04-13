"""
Generating data with simulator_v3 (dependent skills) and fitting a blockbkt model to the data.
Ideally, the model should find good estimates since the data generation and the model match perfectly.
Performs multiple trials and analyzes the results.
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import trange

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
lr = 0.01
epochs = 8
batch_size = 64
step_size = 4
gamma = 0.1

delta = 1
omicron = 0

# simulation setup
np.random.seed(2021)
torch.manual_seed(2021)
simulation_runs = 2
verbose = True
visualize = True
likelihood_check = False

# simulation results
slips = []
guesses = []
l0s = np.zeros((1, simulator_params['n_skills']))
transitions = np.zeros_like(l0s)


if __name__ == '__main__':

    #### SIMULATION
    for _ in trange(simulation_runs):
        # simulate data
        student_simulator = Simulator_v3(**simulator_params, **bkt_params)
        data = student_simulator.sample_students(**sampling_params)

        # initialize dataset
        C = data['correct'].values
        columns = ["skill_name" + str(j) for j in range(n_skills)]
        S = data[columns].values
        user_ids = data['user_id'].values
        train_dataset = BlockBKTDataset(C, S, user_ids)

        # initialize model
        model = BlockBKT(n_skills, bkt_params["max_skills"])

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
            l0s = np.vstack((l0s, l0))
            slips.append(slip)
            guesses.append(guess)
            transitions = np.vstack((transitions, transition))

    
    #### VISUALIZATION
    if visualize:
        # slips
        plt.title('slip')
        plt.hist(slips, bins=100, range=[0, 1])
        plt.axvline(bkt_params['slip'][0], color='red')
        plt.show()

        # guesses
        plt.title('guess')
        plt.hist(guesses, bins=100, range=[0, 1])
        plt.axvline(bkt_params['guess'][0], color='red')
        plt.show()    

        # l0s
        for l in range(l0s.shape[1]):
            plt.title(f'l0 - skill{l}')
            plt.hist(l0s[1:, l], bins=100, range=[0, 1])
            plt.axvline(bkt_params['l0'][l], color='red')
            plt.show()

        # transitions
        for t in range(transitions.shape[1]):
            plt.title(f'transition - skill{t}')
            plt.hist(transitions[1:, t], bins=100, range=[0, 1])
            plt.axvline(bkt_params['transition'][t], color='red')
            plt.show() 


    #### LIKELIHOOD CHECK
    if likelihood_check:
        # insert true transition probabilities
        positive_transitions = torch.tensor(bkt_params['transition']).reshape(-1, 1)
        transition_data = torch.hstack((1-positive_transitions, positive_transitions))
        model.transition.transition_matrix.data = torch.log(transition_data)

        # insert true priors
        positive_priors = torch.tensor(np.array(bkt_params['l0']).reshape(-1, 1))
        prior_data = torch.hstack((1-positive_priors, positive_priors))
        model.priors.data = torch.repeat_interleave(torch.log(prior_data), repeats=n_skills, dim=0)

        # data loader for whole dataset at once (full batch)
        train_loader = torch.utils.data.DataLoader(train_dataset, 
                                                   batch_size=sampling_params['n_students'],
                                                   collate_fn=BlockBKTCollate())

        # compute average negative log-likelihood
        with torch.no_grad():
            for data in train_loader:
                C = data['C'].to(device)
                T = data['T'].to(device)
                S = data['S'].to(device) 
                log_likelihoods = model(C, S, T)
                print('Negative log-likelihood ...', -log_likelihoods.mean())
                break
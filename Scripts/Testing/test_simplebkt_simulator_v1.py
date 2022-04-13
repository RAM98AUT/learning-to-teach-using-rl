"""
Simulating data with simulator (version 1) and fitting a simplebkt model to the data.
Performing multiple trials and analyzing the performance of the model in finding the true data generating parameters.
"""
from torchbkt import *
from simulator import Simulator
from scipy.stats import skew
import torch
import numpy as np
import matplotlib.pyplot as plt


# optimization parameters
lr = 0.05
epochs = 15
batch_size = 16

# performance parameters
bkt_params = {
    'l0': 0.05,
    'transition': 0.2,
    'slip': 0.15,
    'guess': 0.3
}

# data size parameters
n_students = 2000
n_exercises = 50

# experiment setup
trials = 10
verbose = False
np.random.seed(2021)
torch.manual_seed(2021)

# simulation results
l0_results = []
transition_results = []
slip_results = []
guess_results = []

# evaluation functions
def mse_score(y, yhat):
    return np.mean(np.square(yhat-y))

def visualize_results(estimates, true_param):
    plt.hist(estimates, bins=len(estimates))
    plt.axvline(x=true_param, color='red')
    plt.xlim(0, 1)
    plt.show()


if __name__=='__main__':

    #### SIMULATION
    for i in range(trials):
        # import and prepare data
        student_simulator = Simulator(n_skills=1, **bkt_params)
        data, _ = student_simulator.sample_students(n_students=n_students, n_exercises=n_exercises)
        C = data['correct'].values.reshape(-1,1)
        user_ids = data['user_id'].values

        # initialize model
        model = BKT()

        # initialize trainer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        trainer = BKTTrainer(model, optimizer, device)

        # train model
        train_dataset = BKTDataset(C, user_ids)
        trainer.fit(train_dataset, epochs=epochs, batch_size=batch_size, verbose=verbose)

        # extract fitted parameters
        l0 = torch.softmax(model.priors, dim=0)[1].item()
        transition = torch.softmax(model.transition.transition_matrix, dim=0)[1,0].item()
        slip = torch.softmax(model.emission.emission_matrix, dim=1)[1,0].item()
        guess = torch.softmax(model.emission.emission_matrix, dim=1)[0,1].item()

        # store them
        l0_results.append(l0)
        transition_results.append(transition)
        slip_results.append(slip)
        guess_results.append(guess) 


    #### ANALYSE ESTIMATES
    l0_results = np.array(l0_results)
    transition_results = np.array(transition_results)
    slip_results = np.array(slip_results)
    guess_results = np.array(guess_results)

    print(f'MSE l0 ... {mse_score(l0_results, bkt_params["l0"])}')
    print(f'MSE transition ... {mse_score(transition_results, bkt_params["transition"])}')
    print(f'MSE slip ... {mse_score(slip_results, bkt_params["slip"])}')
    print(f'MSE guess ... {mse_score(guess_results, bkt_params["guess"])}')

    print(f'MEAN l0 ... {np.mean(l0_results)}')
    print(f'MEAN transition ... {np.mean(transition_results)}')
    print(f'MEAN slip ... {np.mean(slip_results)}')
    print(f'MEAN guess ... {np.mean(guess_results)}')

    print(f'SKEW l0 ... {skew(l0_results)}')
    print(f'SKEW transition ... {skew(transition_results)}')
    print(f'SKEW slip ... {skew(slip_results)}')
    print(f'SKEW guess ... {skew(guess_results)}')

    visualize_results(l0_results, bkt_params['l0'])
    visualize_results(transition_results, bkt_params['transition'])
    visualize_results(slip_results, bkt_params['slip'])
    visualize_results(guess_results, bkt_params['guess'])
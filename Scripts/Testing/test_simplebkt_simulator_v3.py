"""
Testing simulator_v3/simplebkt by simulating data for a single block with just one skill with simulator_v3
and fitting a simplebkt to the data to find the true data generating parameters.
"""
import numpy as np
import torch

from torchbkt import *
from datahelper import *
from simulator import Simulator_v3


simulator_params = {
    'blocks': [1],
    "n_skills": 1 # sum('blocks')
}

bkt_params = {
    'l0': 0,
    'transition': (0.12,),
    'slip': (0.2,),
    'guess': (0.3,),
    'max_skills': 1
}

sampling_params = {
    'n_students': 2000,
    'n_exercises': 60 
}

# optimization parameters
lr = 0.01
epochs = 10
batch_size = 16
step_size = 10
gamma = 0.1
n_skills = simulator_params['n_skills']

# reproducibility seed
np.random.seed(2021)
torch.manual_seed(2021)


if __name__ == '__main__':
    # simulate data
    student_simulator = Simulator_v3(**simulator_params, **bkt_params)
    data = student_simulator.sample_students(**sampling_params)

    # Prepare data for single blocks
    C = data['correct'].values.reshape(-1,1)
    columns = ["skill_name" + str(j) for j in range(n_skills)]
    S = data[columns].values
    user_ids = data['user_id'].values

    # initialize model
    model = BKT()

    # initialize parameters (for comparison with BlockBKT with blocksize 1)
    # model.priors.data = torch.tensor(np.array([0.5, 0.2]))
    # model.transition.transition_matrix.data[:, 0] = torch.tensor(np.array([0.3, 0.2]))
    # model.emission.emission_matrix.data = torch.tensor(np.array([[0.1, 0.2], [0.15, 0.45]]))

    # initialize trainer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam([{'params': model.priors, 'lr': 1*lr},
                                  {'params': model.emission.parameters()},
                                  {'params': model.transition.parameters(), 'lr': 1*lr}], lr=lr)
    trainer = BKTTrainer(model, optimizer, device)

    # train model
    train_dataset = BKTDataset(C, user_ids)
    trainer.fit(train_dataset, epochs=epochs, batch_size=batch_size)

    # print parameter estimates
    with torch.no_grad():
        print(torch.softmax(model.priors, dim=0))
        print(torch.softmax(model.emission.emission_matrix, dim=1))
        print(torch.softmax(model.transition.transition_matrix, dim=0))
"""
This script can be used to fit a BKT model a given datafile with a single skill 
Data can be generated eg with 'produce_figure'
"""
from torchbkt import *
import torch
import pandas as pd


# optimization parameters
lr = 0.01
epochs = 15
batch_size = 8
data_path = 'data/data_slow.csv'


if __name__=='__main__':
    # import and prepare data
    data = pd.read_csv(data_path)
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
    trainer.fit(train_dataset, epochs=epochs, batch_size=batch_size)

    # print fitted params
    priors = torch.softmax(model.priors, dim=0).detach().numpy()
    transition = torch.softmax(model.transition.transition_matrix, dim=0).detach().numpy()
    emission = torch.softmax(model.emission.emission_matrix, dim=1).detach().numpy()
    print("Estimation of model for l0:", priors[1])
    print("Estimation of model for transition:", transition[1,0])
    print("Estimation of model for slip:", emission[1,0], "for guess:", emission[0,1])
"""
Wrapper for fitting a dataset with multiple skills with the simplebkt model.
Fits a separate BKT model for each skill.
Works for input data with the output format of simulator_v2 and simulator_v3.
"""
import pandas as pd
from .data import BKTDataset
from .models import BKT
from .trainer import BKTTrainer
import torch


class BKTFitMultiple:
    """Wrapper for fitting multiple skills individually.

    Trains a separate BKT model for every skill in the given dataset.

    Attributes
    ----------
    data : pd.DataFrame
        Output of one of the simulators.
    num_skills : int
        Number of distinct skills in data.
    max_num_skills : int (default=2)
        The maximum number of skills within a single exercise.
    lr : float (default=0.01)
        The learning rate for the optimizer.
    epochs : int (default=10)
        The number of epochs to train each individual model.
    batch_size : int (default=8)
    simulator_version : int (default=2)
        Has to be in {2, 3} indicating which simulator generated the data.
    """

    def __init__(self, data, num_skills, max_num_skills=2, lr=0.01, epochs=10, batch_size=8, simulator_version=2):
        """Inits the BKTFitMultiple class with all its attributes."""
        self.data = data
        self.num_skills = num_skills
        self.max_num_skills = max_num_skills
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.simulator_version = simulator_version

    def fitting_multiple(self, verbose=True):
        """Fits one model per skill.
        
        Arguments
        ---------
        verbose : bool (default=True)
            Should epoch training loss information be printed? 

        Returns
        -------
        params : pd.DataFrame
            Dataframe with columns ['P_l0', 'P_T', 'P_guess', 'P_slip'].
            Row 0 corresponds to skill 0, row 1 to skill 1, ...
        """ 
        paramlist = list()

        for skill in range(self.num_skills):
            # prepare data
            if self.simulator_version == 2:
                C, user_ids = self._prepare_data(skill)
            else:
                C, user_ids = self._prepare_data_v3(skill)
            # initialize model
            model = BKT()
            # initialize trainer
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            trainer = BKTTrainer(model, optimizer, device)
            # train model
            train_dataset = BKTDataset(C, user_ids)
            trainer.fit(train_dataset, epochs=self.epochs, batch_size=self.batch_size, verbose=verbose)
            paramlist.append([
                torch.softmax(model.priors, dim=0)[1].item(), # p(L0=1)
                torch.softmax(model.transition.transition_matrix, dim=0)[1, 0].item(), # p(T)
                torch.softmax(model.emission.emission_matrix, dim=1)[0, 1].item(), # p(G)
                torch.softmax(model.emission.emission_matrix, dim=1)[1, 0].item()]) # p(S)

        return pd.DataFrame(paramlist, columns=['P_l0', 'P_T', 'P_guess', 'P_slip'])

    def _prepare_data(self, skill):
        """Prepares simulator_v2 data of a single skill for being used in a BKTDataset.

        Arguments
        ---------
        skill : int
            The skill for which to extract and prepare the data.

        Returns
        -------
        (C, user_ids) : ((n_obs, 1) ndarray, (n_obs,) ndarray) tuple
        """
        # Filter data according to skill
        skill_filter = (self.data["skill_name0"] == skill)
        for i in range(1, self.max_num_skills):
            skill_filter = skill_filter | (self.data["skill_name" + str(i)] == skill)
        skill_data = self.data[skill_filter]
        # Prepare data
        C = skill_data['correct'].values.reshape(-1, 1)
        user_ids = skill_data['user_id'].values
        return C, user_ids

    def _prepare_data_v3(self, skill):
        """Like _prepare_data but for simulator_v3 data."""
        # Filter data according to skill
        skill_filter = self.data['skill_name' + str(skill)] == 1
        skill_data = self.data[skill_filter]
        # Prepare data
        C = skill_data['correct'].values.reshape(-1, 1)
        user_ids = skill_data['user_id'].values
        return C, user_ids
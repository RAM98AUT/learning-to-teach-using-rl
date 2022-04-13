"""
Provides a Trainer class for the BlockBKT model taking care of optimizing the model parameters.
Also provides a wrapper for fitting multiple blocks with the BlockBKTTrainer.
"""
import torch
import pickle
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold
from math import ceil
from tqdm import tqdm

from .data import BlockBKTDataset, BlockBKTCollate
from .models import BlockBKT


class BlockBKTTrainer:
    """ Class for training and validating a BKT model.

    Attributes
    ----------
    model : BKT instance
    optimizer : optimizer 
        From torch.optim tracking model.parameters().
    scheduler : scheduler
        From torch.optim tracking optimizer.
    device : str
        Typically 'cpu' or 'cuda:0'.
    delta : float (default=0.01)
        Controlling the prior regularization to have lower prior probabilities.
    omicron : float (default=0.01)
        Controlling the prior l2-regularization.
    weighted : bool (default=False)
        Whether or not the likelihood for training and validation should be reweighted by the sequence length.
    train_losses : list (init=None)
        Batch losses from fitting the model.
    valid_losses : list (init=None)
        Batch losses on valid_dataset (if provided).
    """

    def __init__(self, model, optimizer, scheduler, device, delta=0.01, omicron=0.01, weighted=False):
        """Inits the BKTTrainer class with model, optimizer, scheduler, device, delta, omicron and weighted."""
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.delta = delta
        self.omicron = omicron
        self.weighted = weighted

        self.model.to(device)
        self.train_losses = None
        self.valid_losses = None

    def fit(self, train_dataset, epochs, batch_size, shuffle=True, verbose=True, valid_dataset=None):
        """Fits the model to a dataset.

        Does not return, but prints epoch loss information if verbose=True.

        Arguments
        ---------
        train_dataset : BKTDataset
        epochs : int
            Number of passes through train_dataset.
        batch_size : int
            Batch size for training.
        shuffle : bool (default=True)
            Whether the samples in the train_loader should be shuffled.
        verbose : bool (default=True)
            Should epoch loss information be printed?
        valid_dataset : BKTDataset (default=None)
        """
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=BlockBKTCollate())
        if valid_dataset is not None:
            valid_loader = DataLoader(valid_dataset,
                                      batch_size=len(valid_dataset),
                                      collate_fn=BlockBKTCollate())

        self.train_losses = [self._validate(train_loader)]
        self.valid_losses = [self._validate(valid_loader)] if valid_dataset is not None else None

        for e in range(epochs):
            # training
            step_losses = self._train_one_epoch(train_loader)
            self.train_losses.append(step_losses)
            self.scheduler.step()
            # validation
            if valid_dataset is not None:
                valid_losses = self._validate(valid_loader)
                self.valid_losses.append(valid_losses)
            # print epoch summary
            if verbose:
                if valid_dataset is not None:
                    print(f'Epoch: {e} ... train loss: {step_losses.mean()} ... valid loss: {valid_losses.mean()}')
                else:
                    print(f'Epoch: {e} ... train loss: {step_losses.mean()}')

    def _train_one_epoch(self, train_loader):
        """Iterates over the train_loader a single time to perform a training epoch.

        Arguments
        ---------
        train_loader : Dataloader

        Returns
        -------
        step_losses : (n_batches,) ndarray
            Array of batch losses for all batches in the epoch.
        """
        step_losses = []
        self.model.train()
        for data in train_loader:
            step_loss = self._train_one_step(data)
            step_losses.append(step_loss)
        return np.array(step_losses)

    def _train_one_step(self, data):
        """Performs training for one given patch.

        Arguments
        ---------
        data : dict
            Dictionary of the form {'C': (batch_size, T_max), 'S': (batch_size, T_max, n_skills), 'T': (batch_size,)}
            as returned by BlockBKTCollate.

        Returns
        -------
        batch_loss : float
        """
        C = data['C'].to(self.device)
        T = data['T'].to(self.device)
        S = data['S'].to(self.device)
        log_likelihoods = self.model(C, S, T, self.weighted)
        penalty = self.model.penalty(self.delta, self.omicron)
        loss = -log_likelihoods.mean() + penalty
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _validate(self, dataloader):
        """Performs one epoch of validation on the given dataloader.

        Arguments
        ---------
        dataloader : Dataloader
            Train or validation dataset to score the current model on.

        Returns
        -------
        epoch_loss : float
            The average batch_losses on the validation dataset.
        """
        self.model.eval()
        with torch.no_grad():
            batch_losses = []
            for data in dataloader:
                C = data['C'].to(self.device)
                T = data['T'].to(self.device)
                S = data['S'].to(self.device)
                log_likelihoods = self.model(C, S, T, self.weighted)
                penalty = self.model.penalty(self.delta, self.omicron)
                loss = -log_likelihoods.mean() + penalty
                batch_losses.append(loss.item())
        return np.array(batch_losses)


class BlocksTrainer:
    """Class to train a collection of blocks (skillbuilder).

    Also includes functionality to track the training and evaluation losses.
    Loops over blocks to fit the BlockBKT model separately for each block.
    Within each block, cross-validation is performed, where all k models are preserved.
    The cross-validation is grouped by user_id, i.e. each student only appears in one single fold.

    With the fit method, the final fit (without cross-validation) can be performed.
    In this case, less information (e.g. less models) will be stored compared to the cross-validation setting.

    Attributes
    ----------
    lr : float (default=0.1)
        The learning rate.
    max_batch_size : int (default=8)
        The batch size that is used is actually ceil(min(max_batch_size, dataset-length/10)).
        This strategy ensures there will not be too little update steps per epoch, and therefore at least some variability.
    min_epochs : int (default=5)
        The minimum number of epochs.
    min_steps : int (default=1000)
        The minimum total number of update steps. Smaller dataset will thus use more training epochs.
    lr_steps : int (default=2)
        The number of learning-rate-decay steps to take. E.g. lr_steps=2 and 15 epochs means that there will be a decay every 5 epochs.
    gamma : float (default=0.1)
        The step multiplier for the learning rate scheduler.
    delta : float (default=0) 
        Controlling the prior regularization to have lower prior probabilities.
    omicron : float (default=0)
        Controlling the transition regularization.
    weighted : bool (default=True)
        Whether or not the likelihood for training and validation should be reweighted by the sequence length.
    device : torch.device (default=torch.device('cpu'))
    models : list (init=[])
        Will contain the n_blocks*n_splits models in a list (len=n_blocks) of lists (len=n_splits each).
    train_learning_curves : list (init=[])
        List of n_blocks lists, where each list contains n_splits numpy arrays containing the batch-losses for the training of each fold.
    valid_learning_curves : list (init=[])
        Same as train_learning_curves but for validation.
    train_lengths : list (init=[])
        List of n_blocks lists, where each list contains again n_splits lists with the sequence lengths of all training samples for that block.
    valid_lengths : list (init=[])
        Same as train_lengths but for the validation samples.
    train_loglikelihoods : list (init=[])
        Same as train_lengths but with the log-likelihoods instead of the sequence lengths
    valid_loglikelihoods : list (init=[])
        Same as train_loglikelihoods but for the validation samples.
    """

    def __init__(self, lr=0.1, max_batch_size=8, min_epochs=5, min_steps=1000, lr_steps=2, gamma=0.1,
                 delta=0, omicron=0, weighted=True, device=torch.device('cpu')):
        """Inits the BlocksTrainer with lr, max_batch_size, min_steps, lr_steps, gamma, delta, omicron, weighted and device."""
        self.lr = lr
        self.max_batch_size = max_batch_size
        self.min_epochs = min_epochs
        self.min_steps = min_steps
        self.lr_steps = lr_steps
        self.gamma = gamma
        self.delta = delta
        self.omicron = omicron
        self.weighted = weighted
        self.device = device
        # init result lists
        self.models = []
        self.train_learning_curves = []
        self.valid_learning_curves = []
        self.train_lengths = []
        self.valid_lengths = []
        self.train_loglikelihoods = []
        self.valid_loglikelihoods = []

    def cross_validate(self, blocks, block_dfs, n_splits=5, verbose=True):
        """Does the actual fitting for a collection of blocks.

        Arguments
        ---------
        blocks : Blocks
            A Blocks instance containing all blocks to be fitted.
        block_dfs : list
            A list containing all block dataframes. len(block_dfs) has to be equal to len(blocks).
        n_splits : int (default=5)
            The number of cross-validation splits.
        verbose : bool (default=True)
            Should epoch loss information be printed?
        """
        for (block, block_df) in tqdm(zip(blocks, block_dfs)):
            # extract data
            C = block_df['correct'].values
            columns = [col for col in block_df.columns if col.startswith('skill_name')]
            S = block_df[columns].values
            user_ids = block_df['user_id'].values
            # perform cross-validation
            block_results = self._init_block_results()
            group_kfold = GroupKFold(n_splits)
            for (train_idx, valid_idx) in group_kfold.split(block_df, groups=user_ids):
                train_dataset = BlockBKTDataset(C[train_idx], S[train_idx], user_ids[train_idx])
                valid_dataset = BlockBKTDataset(C[valid_idx], S[valid_idx], user_ids[valid_idx])
                model = BlockBKT(n_skills=len(block), max_skills=S.sum(axis=1).max())
                fold_results = self._train(model, train_dataset, valid_dataset, verbose)
                block_results = self._update_block_results(block_results, fold_results)
            # store results for current block in instance lists
            self._store_block_results(block_results, cross_validation=True)

    def fit(self, blocks, block_dfs, verbose=True):
        """Performs the final fit, i.e. without cross-validation.
        
        Arguments
        ---------
        blocks : Blocks
            A Blocks instance containing all blocks to be fitted.
        block_dfs : list
            A list containing all block dataframes. len(block_dfs) has to be equal to len(blocks).
        verbose : bool (default=True)
            Should epoch loss information be printed?
        """
        for (block, block_df) in tqdm(zip(blocks, block_dfs)):
            # extract data
            C = block_df['correct'].values
            columns = [col for col in block_df.columns if col.startswith('skill_name')]
            S = block_df[columns].values
            user_ids = block_df['user_id'].values
            train_dataset = BlockBKTDataset(C, S, user_ids)
            # perform training
            model = BlockBKT(n_skills=len(block), max_skills=S.sum(axis=1).max())
            block_results = self._train(model, train_dataset, None, verbose)
            self._store_block_results(block_results, cross_validation=False)

    def _train(self, model, train_dataset, valid_dataset=None, verbose=True):
        """Trains one model using the BlockBKTTrainer class.

        Arguments
        ---------
        model : BlockBKT
        train_dataset : BlockBKTDataset
        valid_dataset : BlockBKTDataset (default=None)
            If provided, the final model will be validated also on the valid_dataset.
        verbose : bool (default=True)
            Should epoch loss information be printed?

        Returns
        -------
        trainer : BlockBKTTrainer
            The Trainer containing the fitted model.
        valid_lengths : list (optional)
            A list containing all sequence lengths of the valid_dataset.
        valid_loglikelihoods : list (optional)
            A list containing all log-likelihoods (not negative!) of the valid_dataset w.r.t. the final model.
        train_lengths : list
            A list containing all sequence lengths of the train_dataset.
        train_loglikelihoods : list
            A list containing all log-likelihoods (not negative!) of the train_dataset w.r.t. the final model.
        """
        # training preparation
        batch_size = ceil(min(self.max_batch_size, len(train_dataset) / 10))
        epochs = ceil(max(self.min_epochs, batch_size * self.min_steps / len(train_dataset)))
        step_size = ceil(epochs / (self.lr_steps + 1))
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=self.gamma)
        # training
        trainer = BlockBKTTrainer(model, optimizer, scheduler, self.device, self.delta, self.omicron, self.weighted)
        trainer.fit(train_dataset, epochs=epochs, batch_size=batch_size, verbose=verbose, valid_dataset=valid_dataset)
        # final sample-wise evaluation
        train_lengths, train_loglikelihoods = self._validate(model, train_dataset, len(train_dataset))
        if valid_dataset is None:
            return trainer, train_lengths, train_loglikelihoods
        else:
            valid_lengths, valid_loglikelihoods = self._validate(model, valid_dataset, len(valid_dataset))
            return trainer, valid_lengths, valid_loglikelihoods, train_lengths, train_loglikelihoods

    def _validate(self, model, dataset, batch_size):
        """Evaluates a trained model on a dataset.

        Computes the log-likelihoods for all samples.

        Arguments
        ---------
        model : BlockBKT
            A trained BlockBKT model.
        dataset : BlockBKTDataset
            A BlockBKTDataset compatible with model.
        batch_size : int
        weighted : bool (default=False) 
            Should the likelihoods be weighted by the sequence lengths.

        Returns
        -------
        lengths : list
            A list containing all sequence lengths of the dataset.
        log_likelihoods : list
            A list containing all log-likelihoods (not negative!) of the dataset w.r.t. the model. 
        """
        data_loader = DataLoader(dataset, batch_size, collate_fn=BlockBKTCollate())
        model.eval()
        lengths = []
        log_likelihoods = []
        with torch.no_grad():
            for data in data_loader:
                c = data['C'].to(self.device)
                t = data['T'].to(self.device)
                s = data['S'].to(self.device)
                log_liks = model(c, s, t, weighted=self.weighted)
                log_likelihoods.extend(log_liks.tolist())
                lengths.extend(t.tolist())
        return lengths, log_likelihoods

    def _init_block_results(self):
        """Initializes and returns a dictionary of lists to store all cross-val information for one block."""
        block_results = {
            'models': [],
            'train_learning_curves': [],
            'valid_learning_curves': [],
            'train_lengths': [],
            'valid_lengths': [],
            'train_loglikelihoods': [],
            'valid_loglikelihoods': []
        }
        return block_results

    def _update_block_results(self, block_results, fold_results):
        """Updates the cross-val information for one block with the current fold's results."""
        trainer, valid_lengths, valid_loglikelihoods, train_lengths, train_loglikelihoods = fold_results
        block_results['models'].append(trainer.model)
        block_results['train_learning_curves'].append(trainer.train_losses)
        block_results['valid_learning_curves'].append(trainer.valid_losses)
        block_results['valid_lengths'].append(valid_lengths)
        block_results['valid_loglikelihoods'].append(valid_loglikelihoods)
        block_results['train_lengths'].append(train_lengths)
        block_results['train_loglikelihoods'].append(train_loglikelihoods)
        return block_results

    def _store_block_results(self, block_results, cross_validation=False):
        """Adds all the block information and results to the respective instance attributes.
        Ensures the same format for training and cross-val even though cross-val will store more elements.
        """
        if cross_validation:
            self.models.append(block_results['models'])
            self.train_learning_curves.append(block_results['train_learning_curves'])
            self.valid_learning_curves.append(block_results['valid_learning_curves'])
            self.valid_lengths.append(block_results['valid_lengths'])
            self.valid_loglikelihoods.append(block_results['valid_loglikelihoods'])
            self.train_lengths.append(block_results['train_lengths'])
            self.train_loglikelihoods.append(block_results['train_loglikelihoods'])
        else:
            trainer, train_lengths, train_loglikelihoods = block_results
            self.models.append([trainer.model])
            self.train_learning_curves.append([trainer.train_losses])
            self.train_lengths.append([train_lengths])
            self.train_loglikelihoods.append([train_loglikelihoods])

    def save(self, filename):
        """Saves the instance as pickle file.
        
        Arguments
        ---------
        out : str
            Filename (including path) where to store the file.
        """
        with open(filename, 'wb') as out:
            pickle.dump(self, out)
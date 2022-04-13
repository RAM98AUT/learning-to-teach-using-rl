"""
Provides a Trainer class for the BKT model taking care of optimizing the model parameters.
Also prints loss information for every training epoch.
"""
from torch.utils.data import DataLoader
from .data import BKTCollate


class BKTTrainer:
    """Class for training a BKT model.

    Attributes
    ----------
    model : BKT instance
    optimizer : optimizer 
        From torch.optim tracking model.parameters().
    device : str
        Typically 'cpu' or 'cuda:0'.
    epoch_losses : list or None
        Epoch losses from fitting the model.
    """

    def __init__(self, model, optimizer, device):
        """Inits the BKTTrainer class with args model, optimizer and device."""
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
        self.epoch_losses = None

    def fit(self, train_dataset, epochs, batch_size, shuffle=True, verbose=True):
        """Fits the model to a dataset.

        Arguments
        ---------
        train_dataset : BKTDataset instance
        epochs : int
            Number of passes through train_dataset.
        batch_size : int
            Batch size for training.
        shuffle : bool (default=True)
            Should the data be shuffled before every epoch.
        verbose : bool (default=True)
            Should epoch loss information be printed?
        """
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  collate_fn=BKTCollate())
        self.epoch_losses = []
        for e in range(epochs):
            epoch_loss = self._train_one_epoch(train_loader)
            self.epoch_losses.append(epoch_loss)
            if verbose:
                print(f'Epoch: {e} ... train loss: {epoch_loss}')

    def _train_one_epoch(self, train_loader):
        """Trains the model for one epoch.

        Arguments
        ---------
        train_loader : Dataloader
            From torch.utils.data providing the train_dataset.

        Returns
        -------
        epoch_loss : float
            The average training loss for the current epoch.        
        """
        step_losses = []
        self.model.train()
        for data in train_loader:
            step_loss = self._train_one_step(data)
            step_losses.append(step_loss)
        return sum(step_losses) / len(step_losses)

    def _train_one_step(self, data):
        """Trains the model for one batch making one gradient descent update step.
        
        Arguments
        ---------
        data : dict
            Dictionary of the form {'C': (batch_size, T_max), 'T': (batch_size,)}
            as returned by BKTCollate.
    
        Returns
        -------
        loss : float
            The mean negative log-likelihood of the batch.
        """
        C = data['C'].to(self.device)
        T = data['T'].to(self.device)
        log_likelihoods = self.model(C, T)
        loss = -log_likelihoods.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
"""
Implements a dataset class to feed the BKT model with data in the correct form.
Also takes care of padding sequences of different length. (see: BKTCollate)
"""
import numpy as np
import torch


class BKTDataset:
    """Implements a dataset class for feeding a BKT model with data.

    The implementation works for a single skill only.

    Attributes
    ----------
    C : (n_obs, 1) ndarray
        Indicates whether an exercise was answered correctly.
    user_id : (n_obs,) ndarray
        Indicates which user is concerned by each observations.
        The observations must be ordered by time within each user.
    unique_ids : (n_users,) ndarray
        The unique user-ids in the dataset.
    """

    def __init__(self, C, user_ids):
        """Inits the BKTDataset class with the ndarrays C and user_ids.
        
        C can be provided as (n_obs, 1) or (n_obs,) ndarray.
        """
        self.C = C.reshape(-1, 1)
        self.user_ids = user_ids
        self.unique_ids = np.unique(user_ids)

    def __len__(self):
        """Returns the length of the dataset, which corresponds to the number of unique user-ids."""
        return len(self.unique_ids)

    def __getitem__(self, idx):
        """Returns the observation for a given index.

        Arguments
        ---------
        idx : int

        Returns
        -------
        (c, t) : ((1, T), int) tuple
            c is the sequence of True/False answers for the user with index idx.
            t is the length of the sequence, i.e. c.shape[0]. 
        """
        user = self.unique_ids[idx]
        c = self.C[self.user_ids == user, :].T
        t = c.shape[1]
        return (c, t)


class BKTCollate:
    """Class returning a callable for collating BKT observations.

    The __call__ method takes a batch of observations from a BKTDataset.
    It pads them to the same length and concatenates all sequence lenghts.

    Arguments
    ---------
    batch : list
        A list containing observations as returned by the __getitem__ method from BKTDataset.

    Returns
    -------
    dict : {'C': (batch_size, T_max) tensor, 
            'T': (batch_size,) tensor}
        C is the observation matrix with sequences padded to the same length.
        T contains the actual length information for each sequence.
    """

    def __call__(self, batch):
        # pad sequences to same length
        lengths = [x[0].shape[1] for x in batch]
        T_max = max(lengths)
        C = [np.pad(x[0], [(0, 0), (0, T_max - x[0].shape[1])]) for x in batch]
        C = np.vstack(C)
        # sequence length array
        T = np.array([x[1] for x in batch])

        return {
            'C': torch.tensor(C, dtype=torch.int64),
            'T': torch.tensor(T, dtype=torch.int64)
        }
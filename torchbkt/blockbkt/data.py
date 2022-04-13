"""
Implements a dataset class to feed the BlockBKT model with data in the correct form.
Also takes care of padding sequences of different length. (see: BlockBKTCollate)
"""
import numpy as np
import torch


class BlockBKTDataset:
    """Implements a dataset class for feeding a BlockBKT model with data.

    The implementation currently works for a single skill only.

    Attributes
    ----------
    C : (n_obs,) ndarray 
        Indicates whether an exercise was answered correctly.
    S : (n_obs, n_skills) ndarray
        Indicates which skills were challenged in each exercise.
    user_ids : (n_obs,) ndarray 
        Indicates which user is concerned by each observations.
        The observations must be ordered by time within each user.
    unique_ids : (n_users,) ndarray
        The unique user-ids in the dataset.
    """

    def __init__(self, C, S, user_ids):
        """Inits the BlockBKTDataset class with the ndarrays C, S and user_ids."""
        self.C = C.ravel()
        self.S = S
        self.user_ids = user_ids.ravel()
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
        (c, s, t) : ((1, T), (n_skills, T), int) tuple
            c is the sequence of True/False answers for the user with index idx.
            s is the sequence that defines which skills are present in each exercise
            t is the length of the sequence, i.e. c.shape[0]. 
        """
        user = self.unique_ids[idx]
        c = self.C[self.user_ids == user].reshape(1, -1)
        s = self.S[self.user_ids == user, :].T
        t = c.shape[1]
        return (c, s, t)


class BlockBKTCollate:
    """Class returning a callable for collating BlockBKT observations.

    The __call__ method takes a batch of observations from a BlockBKTDataset.
    It pads them to the same length and concatenates them.

    Arguments
    ---------
    batch : list
        A list containing observations as returned by the __getitem__ method from BlockBKTDataset.

    Returns
    -------
    dict : {'C': (batch_size, T_max) tensor, 
            'S': (batch_size, T_max, n_skills) tensor,
            'T': (batch_size,) tensor}
        C is the observation matrix with sequences padded to the same length.
        S contains the 0-1 encoding indicating which exercise type was asked each time.
        T contains the actual length information for each sequence. 
    """

    def __call__(self, batch):
        # pad sequences to same length
        lengths = [x[2] for x in batch]
        T_max = max(lengths)
        C = [np.pad(x[0], [(0, 0), (0, T_max - x[2])]) for x in batch]
        C = np.vstack(C)
        # calculate S
        S = [self._pad_S(x[1], x[2], T_max).T for x in batch]
        S = np.stack(S, axis=0)
        # sequence length array
        T = np.array(lengths)

        return {
            'C': torch.tensor(C, dtype=torch.int64),
            'S': torch.tensor(S, dtype=torch.int64),
            'T': torch.tensor(T, dtype=torch.int64)
        }

    def _pad_S(self, s, t, T_max):
        """Pads an exercise sequence to a given length.

        Exercise sequences must NOT simply be zero-padded.
        A valid exercise type has to be used.
        Therefore, it is padded with exercise type (1, 0, ..., 0).

        Arguments
        ---------
        s : (n_skills, t) ndarray
            The actually observed exercise sequence.
        t : int
            The actual length of the sequence.
        T_max : int
            The target sequence length.

        Returns
        -------
        s_new : (n_skills, T_max) ndarray
            The sequence s padded to length T_max with exercise (1, 0, ..., 0).
        """
        s = np.pad(s, [(0, 0), (0, T_max - t)])
        add_on = np.zeros_like(s)
        add_on[0, :] = 1
        add_on *= np.all(s==0, axis=0, keepdims=True)
        return s + add_on
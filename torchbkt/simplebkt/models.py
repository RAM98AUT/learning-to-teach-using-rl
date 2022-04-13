"""
Contains the necessary PyTorch modules for the implementation of a HMM for Bayesian Knowledge Tracing with a single skill.
The main class BKT itself contains priors, a TransitionModel and an EmissionModel.
"""
import torch
import torch.nn as nn
import numpy as np


class BKT(nn.Module):
    """PyTorch implementation of a HMM for Bayesian Knowledge Tracing.

    The implementation works for a single skill only.
    Therefore, the number of possible states is just 2, i.e. {unlearned, learned}.

    Attributes
    ----------
    transition : TransitionModel
    emission : EmissionModel
    priors : (2,) parameter 
        Representing the unnormalized initial probabilities (0, 1) of the learning state.
        After applying softmax, self.priors[i] corresponds to P(L0=i) with i in {1, 2}.
    """

    def __init__(self):
        """Inits the BKT class without any args."""
        super(BKT, self).__init__()
        self.transition = TransitionModel()
        self.emission = EmissionModel()
        self.priors = nn.Parameter(torch.rand(2))


class TransitionModel(nn.Module):
    """PyTorch implementation of the internal transition model of a HMM for Bayesian Knowledge Tracing.

    The implementation works for a single skill only.

    Attributes
    ----------
    transition_matrix : (2, 2) parameter 
        Representing the unnormalized transition matrix. 
        To obtain probabilities, apply softmax column-wise.
        Entry (i, j) stands for the transition from state j to state i.
        State 0 stands for unlearned, state 1 stands for learned.

        self.transition_matrix[0, 0] ... 1-p(T)
        self.transition_matrix[0, 1] ... 0 (learned -> unlearned) -> set to -inf due to softmax
        self.transition_matrix[1, 0] ... p(T)
        self.transition_matrix[1, 1] ... 1 (learned -> learned) -> set to 0 due to softmax
    """

    def __init__(self):
        """Inits the TransitionModel class without any args."""
        super(TransitionModel, self).__init__()
        self.transition_matrix = nn.Parameter(torch.rand(2, 2))
        self.transition_matrix.data[0, 1] = -np.inf  # learned -> unlearned
        self.transition_matrix.data[1, 1] = 0  # learned -> learned


class EmissionModel(nn.Module):
    """PyTorch implementation of the emission model of a HMM for Bayesian Knowledge Tracing.

    The implementation works for a single skill only.

    Attributes
    ----------
    emission_matrix : (2, 2) parameter 
        Representing the unnormalized emission matrix. 
        To obtain probabilities, apply softmax row-wise.
        Entry (i, j) stands for the emission of outcome j from state i.
        State 0 stands for unlearned, state 1 stands for learned.
        Outcome 0 stands for an incorrect answer, outcome 1 for a correct one.

        self.emission_matrix[0, 0] ... 1-p(G)
        self.emission_matrix[0, 1] ... p(G)
        self.emission_matrix[1, 0] ... p(S)
        self.emission_matrix[1, 1] ... 1-p(S)
    """

    def __init__(self):
        """Inits the EmissionModel class without any args."""
        super(EmissionModel, self).__init__()
        self.emission_matrix = nn.Parameter(torch.rand(2, 2))
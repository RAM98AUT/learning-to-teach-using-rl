"""
Contains the necessary PyTorch modules for the implementation of a HMM for Bayesian Knowledge Tracing with multiple skills with a block structure.
The main class BlockBKT itself contains priors, a BlockTransitionModel and a BlockEmissionModel.
"""
import torch
import torch.nn as nn

from scipy.special import comb
from itertools import chain, combinations


class BlockBKT(nn.Module):
    """PyTorch implementation of a HMM for Bayesian Knowledge Tracing with a block structure.

    The implementation works for a block of dependent skills.
    The number of possible states is 2^n_skills.
    Each exercise can concern multiple (up to max_skills) skills.
    We assume that a student is capable of solving an exercise if he/she has all relevant skills for that exercise in the learned state.
    Otherwise, there is still a probability of p(guess) that the exercise is solved correctly.

    Attributes
    ----------
    n_skills : int
        The number of skills of the given block.
    max_skills : int
        The maximum number of skills that can be relevant for a single exercise.
    n_states : int
        The number of states (=2**n_skills).
    state_tensor : (n_states, n_skills) tensor (long)
        Holds the encodings of all possble states. 
        Entry (i,j) indicates whether skill j is learned in state i.
    exercise_tensor : (n_exercises, n_skills) tensor (long)
        Subtensor of state_tensor containing all rows that can be an exercise type.
    priors : (n_states,) parameter
        Representing the unnormalized initial probabilities (0, 1) of the states.
    transition : BlockTransitionModel
    emission : BlockEmissionModel
    """

    def __init__(self, n_skills, max_skills):
        """Inits the BlockBKT class with n_skills and max_skills."""
        super(BlockBKT, self).__init__()
        # states and exercise types
        self.n_skills = n_skills
        self.max_skills = max_skills
        self.n_states = 2**n_skills
        self._get_state_tensor()
        self._get_exercise_tensor()
        # parameters
        self.priors = nn.Parameter(torch.rand(self.n_skills, 2))
        self.transition = BlockTransitionModel(
            self.state_tensor, self.exercise_tensor)
        self.emission = BlockEmissionModel()

    def _get_state_tensor(self):
        """Internal method to create the state_tensor."""
        # get statelist
        skill_set = range(self.n_skills)  # [0, 1, 2, ...]
        statelist = [combinations(skill_set, p) for p in range(self.n_skills + 1)] # [[()], [(0,), (1,), ...], [(0, 1), (0, 2), ...], ...]
        statelist = list(chain.from_iterable(statelist)) # [(), (0,), (1,), ..., (0, 1), (0, 2), ..., (1, 2), (1, 3), ...]
        # define indices to overwrite with ones
        state_indices = [[i]*len(skills) for (i, skills) in enumerate(statelist)]
        state_indices = torch.tensor(list(chain.from_iterable(state_indices)), dtype=torch.long)
        skill_indices = torch.tensor([s for skills in statelist for s in skills], dtype=torch.long)
        # get statetensor
        self.state_tensor = torch.zeros((self.n_states, self.n_skills), dtype=torch.long)
        self.state_tensor[(state_indices, skill_indices)] = 1

    def _get_exercise_tensor(self):
        """Internal method to create the exercise tensor.

        It is equal to state_tensor[1:self.n_exercises+1, :].
        I.e. the first state [0, 0, ..., 0] and states with more than max_skills ones are not valid exercises.
        """
        self.n_exercises = sum([int(comb(self.n_skills, k)) for k in range(1, self.max_skills+1)])
        self.exercise_tensor = self.state_tensor[1:(self.n_exercises+1), :].clone()


class BlockTransitionModel(nn.Module):
    """PyTorch implementation of the internal transition step of a HMM for Bayesian Knowledge Tracing with a block structure.

    The implementation works for a block of dependent skills.

    Attributes
    ----------
    exercise_tensor : (n_exercises, n_skills) tensor (long)
        Tensor containing all rows that can be an exercise type.
    transition_matrix : (n_skills, 2) parameter
        Representing the unnormalized transition probabilities.
        To obtain probabilities, apply softmax row-wise.
        Then, column 0 contains 1-p(T) and column 1 contains p(T) for each skill.

        Will be used to compute the (n_states, n_states, n_exercises) transition tensor in the forward method.
        (transition probability from every state to every state for every possible exercise type)
        (transition_tensor[i,j,k] ... p(state(t+1)=i | state(t)=j) if exercise k was assigned at time t.)
    """

    def __init__(self, state_tensor, exercise_tensor):
        """Inits the BlockTransitionModel class with arguments state_tensor and exercise_tensor."""
        super(BlockTransitionModel, self).__init__()
        n_skills = state_tensor.shape[1]
        self.exercise_tensor = exercise_tensor
        self.transition_matrix = nn.Parameter(torch.rand(n_skills, 2))


class BlockEmissionModel(nn.Module):
    """PyTorch implementation of the emission step of a HMM for Bayesian Knowledge Tracing with a block structure.

    The implementation is compatible with the simble BKT and the BlockBKT model.
    I.e. the slip and guess parameters will be shared parameters for all exercise types.

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
        super(BlockEmissionModel, self).__init__()
        self.emission_matrix = nn.Parameter(torch.rand(2, 2))
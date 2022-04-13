"""
SimpleBKT
=========

Implementation of a HMM for Bayesian Knowledge Tracing with independent skills in PyTorch.

The main class BKT contains priors, a TransitionModel and an EmissionModel.
The priors determine the probabilities of the two possible states {0=unlearned, 1=learned} at the beginning of a sequence.
The TransitionModel determines p(T), the transition probability from state 0 to state 1.
By assumption, unlearning, i.e. a transition from state 1 to state 0, is not possible.
The EmissionModel determines the probabilities of the two possible outcomes {0=wrong, 1=correct} for each hidden state when taking an exercise.
It primarily tracks p(S), the probability to slip even though the skill is in the learned state, and p(G), the probability to guess correctly even though the skill is unlearned.

The implementation includes
    - PyTorch models for BKT with a single skill.
    - Corresponding forward methods for computing the log-likelihoods.
    - Dataset and Collate classes for feeding data to the model in the correct shape.
    - Trainer classes for optimizing the HMM parameters with gradient descent.
    - Wrapper class to fit data with multiple skills by training a separate model for each skill.
"""

from .models import BKT, TransitionModel, EmissionModel
from .forward import forward_algorithm, forward_transition, forward_emission
from .data import BKTDataset, BKTCollate
from .trainer import BKTTrainer
from .fit_multiple_skills import BKTFitMultiple


# Define forward methods
BKT.forward = forward_algorithm
EmissionModel.forward = forward_emission
TransitionModel.forward = forward_transition
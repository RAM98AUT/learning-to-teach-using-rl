"""
BlockBKT
========

Implementation of a HMM for Bayesian Knowledge Tracing with dependent skills in PyTorch.

The main class BlockBKT contains priors, a BlockTransitionModel and a BlockEmissionModel.
The implementation works for a block of dependent skills.

The number of possible states is 2^n_skills, e.g. n_skills=3 implies the following possible states:
000
100
010
001
110
101
011
111
Each exercise can concern multiple (up to max_skills) skills.

If in the above example max_skills=2, the set of possible exercises is:
100
010
001
110
101
011

We assume that a student is capable of solving an exercise if he/she has all relevant skills for that exercise in the learned state.
Otherwise, there is still a probability of p(guess) that the exercise is solved correctly.
Example:
In state 100, the probability to solve an exercise of type 110 correctly is p(guess), since the second skill is unlearned.
In state 110, the probability to solve an exercise of type 110 correctly is 1-p(slip), since all relevant skills are learned.
"""

from .models import BlockBKT, BlockTransitionModel, BlockEmissionModel
from .forward import forward_block, forward_block_transition, forward_block_emission
from .forward import _get_gradient, _get_mask, _get_transition_tensor, penalty
from .data import BlockBKTDataset, BlockBKTCollate
from .trainer import BlockBKTTrainer, BlocksTrainer


# Define forward methods
BlockBKT.forward = forward_block
BlockEmissionModel.forward = forward_block_emission
BlockTransitionModel.forward = forward_block_transition

# Define helper methods
BlockBKT._get_gradient = _get_gradient
BlockBKT._get_mask = _get_mask
BlockBKT._get_transition_tensor = _get_transition_tensor
BlockBKT.penalty = penalty
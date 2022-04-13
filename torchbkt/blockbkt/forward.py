"""
Contains the forward methods for the models defined in models.py, i.e. BlockBKT, BlockTransitionModel and BlockEmissionModel.
"""
import numpy as np
import torch
import torch.nn.functional as F


def forward_block(self, C, S, T, weighted=False):
    """Implements the forward algorithm of a HMM for blockwise Bayesian Knowledge Tracing.

    Computes the log-likelihoods for a batch of observed student sequences.

    Arguments
    ---------
    C : (batch_size, T_max) tensor
        Holds one sequence in every row, each padded to length T_max.
    S : (batch_size, T_max, n_skills) tensor
        Holds the 0-1 encoding of the assigned exercise types.
    T : (batch_size,) tensor
        Holds the actual (unpadded) length information for the sequences.
    weighted : bool (default=False)
        If True, the log-likelihoods will be divided by the sequence lenghts.
        This strategy is supposed to counteract the problem of variable sequence lenghts in training and evaluation.

    Returns
    -------
    log-likelihoods : (batch_size, 1) tensor
        Holds the log-likelihoods of all observations in the batch.
    """
    # handle dimensions
    batch_size = C.shape[0]
    T_max = C.shape[1]
    state_tensor = torch.repeat_interleave(self.state_tensor.unsqueeze(-1), batch_size, dim=2)
    # init transitions
    log_transitions = self._get_transition_tensor()
    # init priors
    priors = torch.softmax(self.priors, dim=1)
    state_priors = priors[0, self.state_tensor[:, 0]]
    for i in range(1, self.n_skills):
        state_priors = torch.mul(state_priors, priors[i, self.state_tensor[:, i]])
    log_priors = torch.log(state_priors)
    # alpha propagation
    log_alphas = torch.zeros(batch_size, T_max, self.n_states)
    log_alphas[:, 0, :] = self.emission(C[:, 0], S[:, 0, :], state_tensor) + log_priors
    for t in range(1, T_max):
        log_alphas[:, t, :] = self.emission(C[:, t], S[:, t, :], state_tensor) + self.transition(S[:, t-1, :], log_alphas[:, t-1, :], log_transitions)
    # collect final log-likelihoods
    log_sums = torch.logsumexp(log_alphas, dim=2)
    log_likelihoods = torch.gather(log_sums, dim=1, index=(T-1).view(-1, 1))
    return log_likelihoods / T.reshape(-1, 1) if weighted else log_likelihoods


def forward_block_emission(self, c, s, state_tensor):
    """Implements a blockwise emission forward step.

    Given the outcomes for one timestep c_t and a given exercise s_t, computes log(p(c_t|l_t=0)) and log(p(c_t|l_t=1)).
    l_t=1 means that all skills challenged by exercise s_t are in the learned state, l_t=0 is the complementary case.

    Arguments
    ---------
    c : (batch_size,) tensor
        Holds the outcomes for a particular time-step for the batch.
    s : (batch_size, n_skills) tensor
        Holds the 0-1 encoded skills for all observations for the given timestep.
    state_tensor : (n_states, n_skills, batch_size) tensor
        The state_tensor from BlockBKT stacked batch_size times in dim 2.

    Returns
    -------
    (batch_size, n_states) tensor 
        Holding the conditional log-probabilities of the outcomes for each state for each sample.
    """
    # find out which state-exercise combinations are in learned/unlearned state
    comparison = state_tensor >= s.transpose(0, 1).unsqueeze(0)
    learned = torch.all(comparison, dim=1)
    unlearned = torch.logical_not(learned)
    learned_tensor = torch.stack((unlearned.float(), learned.float()), dim=2)
    # compute (log) probs for both possible outcomes
    log_emission = torch.log_softmax(self.emission_matrix, dim=1)
    log_probs = learned_tensor @ log_emission
    # filter and return log probs of observed outcomes
    filter_mask = torch.repeat_interleave(
        c.reshape(1, -1), state_tensor.shape[0], 0).unsqueeze(-1)
    log_probs = torch.gather(log_probs, dim=2, index=filter_mask)
    return torch.squeeze(log_probs, dim=2).transpose(0, 1)


def forward_block_transition(self, s, log_alpha, log_transitions):
    """Implements a transition forward step.

    Given the log-alphas from the last timestep and the just completed exercise type, computes the logsumexp over all states of the log transition matrix + the old log-alphas. 

    Arguments
    ---------
    s : (batch_size, n_skills) tensor
        The exercise type that was just completed by each student.
    log_alpha : (batch_size, n_states) tensor
        Holds log-alphas for all sequences of the preceding timestep.

    Returns
    -------
    (batch_size, n_states) tensor 
        Holds the transition part for the computation of the next log-alphas.
    """
    s_indices = torch.argmax(torch.all(self.exercise_tensor.unsqueeze(-1)==s.transpose(0, 1).unsqueeze(0), dim=1).float(), dim=0)
    log_transitions = log_transitions[:, :, s_indices]
    log_alpha = log_alpha.transpose(0, 1).unsqueeze(0)
    return torch.logsumexp(log_transitions + log_alpha, dim=1).transpose(0, 1)


def _get_gradient(self):
    """Internal method to compute the "gradient" for all ordered state combinations.

    The gradient from state [0,0,1] to state [1,0,1] is [1,0,0] for instance.
    A transition is generally impossible if any entry of the gradient is -1.
    A transition is only possible with respect to a specific exercise type, if exercise >= gradient (elementwise >=).
    A transition from state [0,0,1] to state [1,0,1] (gradient is [1,0,0]) is not possible with exercise [0,1,0].

    Returns
    -------
    gradient : (n_states, n_states, n_skills) tensor
      gradient[i,j,:] holds the gradient from state j to state i.
    """
    target = torch.repeat_interleave(self.state_tensor.unsqueeze(1), self.n_states, dim=1)
    origin = self.state_tensor.unsqueeze(0)
    return target - origin


def _get_mask(self, gradient):
    """Creates a mask to mask impossible transitions in the transition tensor.

    All entries in the mask are in {0, 0.5, 1}.
    The mask will be used to mask the transition tensor, i.e. set all impossible transitions to -inf.

    Arguments
    ---------
    gradient : (n_states, n_states, n_skills) tensor
        The gradient as returned by _get_gradient.

    Returns
    -------
    mask : (n_states, n_states, n_exercises) tensor
        Is 1 in all positions with a valid transition.
        Is 0 or 0.5 in all other positions.
        Can be used as torch.where(mask < 1, ...).
    """
    n_exercises = self.exercise_tensor.shape[0]
    # impossible transitions due to no 'unlearning' (n_states, n_states)
    infeasibility_mask = torch.all(gradient >= 0, dim=2).float()
    # transition not possible for specific exercises (n_states, n_states, n_exercises)
    gradient_stack = torch.repeat_interleave(gradient.unsqueeze(-1), repeats=n_exercises, dim=3)
    exercise_stack = self.exercise_tensor.transpose(0, 1).unsqueeze(0).unsqueeze(0)
    exercise_mask = torch.all(exercise_stack >= gradient_stack, dim=2).float()
    # combine and return the masks
    return (exercise_mask + infeasibility_mask.unsqueeze(-1)) / 2


def _get_transition_tensor(self):
    """Computes the log transition tensor.
    
    log_transition_tensor[i, j, k] = log(p(state[t+1]=i | state[t]=j)) if exercise[t]=k.
    All events having probability 0 therefore have value -inf in the corresponding cell.

    Returns
    -------
    transition_tensor : (n_states, n_states, n_exercises) tensor
        Contains the log transition probabilities for all state combinations for each exercise type.
    """
    # get the log transition-probabilities for all skills
    log_probs = F.log_softmax(self.transition.transition_matrix, dim=1)
    # stack the above matrix in dimension 2 (one matrix for each exercise type)
    log_probs = torch.repeat_interleave(log_probs.unsqueeze(-1), self.exercise_tensor.shape[0], dim=2)

    # see: _get_gradient docstring
    gradient = self._get_gradient()
    mask = self._get_mask(gradient)
    # set all columns where skill was already learned to zero
    learned_cols = torch.all(gradient <= 0, dim=0, keepdim=True)
    gradient = torch.where(learned_cols, -1, gradient)

    # exercise tensor in different shape: encoded exercise types as col vectors stacked behind one another
    exercise_mask = self.exercise_tensor.transpose(0, 1).unsqueeze(1)
    # zero out probabilities that are irrelevant for each exercise type
    probs_mask = log_probs * exercise_mask

    # extract only positions where a positive transition would take place from gradient
    gradient_positive = torch.where(gradient > 0, gradient, 0)
    # multiply that with the corresponding log-transition probabilities (sum up transition probabilities in log-space)
    positive_result = torch.matmul(gradient_positive.float(), probs_mask[:, 1, :])

    # extract all positions where no transition or negative transition (impossible) would take place from gradient
    gradient_negative = torch.where(gradient == 0, 1, 0)
    # multiply that with the corresponding log counter-transition-probabilities (sum up counter transition probabilities in log-space)
    negative_result = torch.matmul(gradient_negative.float(), probs_mask[:, 0, :])

    # add up transition and counter probabilities in log-space
    result = positive_result + negative_result
    # mask impossible and irrelevant transitions
    transition_tensor = torch.where(mask < 1, -np.inf*torch.ones_like(result), result)

    return transition_tensor


def penalty(self, delta=0, omicron=0):
    """Regularization for prior and transition parameters.

    The first regularization term tries to keep the prior-probabilities low by using the following term:
      delta * [(mean of all positive unnormalized prior parameters) - (mean of all negative unnormalized prior parameters)]

    The second regularization term tries to keep the transition-probabilities low by using the following term:
      delta * [(mean of all positive unnormalized transition parameters) - (mean of all negative unnormalized transition parameters)]

    In both terms, l2-regularization is added.
    In [-1, 1], l2-regularization is weak compared the above regularization term.
    Outside of this interval, it gets more influential.

    Arguments
    ---------
    delta : float (default=0.01)
      Controls the prior regularization term.
    omicron : float (default=0.01)
      Controls the transition regularization term.
    """
    prior_means = self.priors.mean(dim=0)
    transition_means = self.transition.transition_matrix.mean(dim=0)
    prior_penalty = delta * (prior_means[1] - prior_means[0] + self.priors.square().mean())
    transition_penalty = omicron * (transition_means[1] - transition_means[0] + self.transition.transition_matrix.square().mean())
    return prior_penalty + transition_penalty
"""
Contains the forward methods for the models defined in models.py, i.e. BKT, TransitionModel and EmissionModel.
"""
import torch
import torch.nn.functional as F


def forward_algorithm(self, C, T):
    """Implements the forward algorithm of a HMM for Bayesian Knowledge Tracing.

    Computes the log-likelihoods for a batch of observed student sequences.

    Arguments
    ---------
    C : (batch_size, T_max) tensor
        Holds one sequence in every row, each padded to length T_max.
    T : (batch_size,) tensor
        Holds the actual (unpadded) length information for the sequences.

    Returns
    -------
    log-likelihoods : (batch_size, 1) tensor
        Holds the log-likelihoods of all sequences in the batch.
    """
    # handle dimensions
    batch_size = C.shape[0]
    T_max = C.shape[1]

    # initialize forward pass
    log_priors = F.log_softmax(self.priors, dim=0)
    log_alphas = torch.zeros(batch_size, T_max, 2)

    # alpha propagation
    log_alphas[:, 0, :] = self.emission(C[:, 0]) + log_priors
    for t in range(1, T_max):
        log_alphas[:, t, :] = self.emission(
            C[:, t]) + self.transition(log_alphas[:, t-1, :])

    # compute log-likelihoods from final alphas (solution from reference)
    log_sums = torch.logsumexp(log_alphas, dim=2)
    log_likelihoods = torch.gather(log_sums, dim=1, index=(T-1).view(-1, 1))
    return log_likelihoods


def forward_emission(self, c):
    """Implements an emission forward step.

    Given the outcomes for one timestep c_t, computes log(p(c_t|l_t=0)) and log(p(c_t|l_t=1)).
    c_t in {0, 1} is the outcome at timestep t, l_t is the learning state at timestep t.

    Arguments
    ---------
    c : (batch_size,) tensor
        Holds the outcomes for a particular time-step for the batch.

    Returns
    -------
    (batch_size, 2) tensor 
        Holding the conditional log-probabilities of the outcomes for both possible learning states.
    """
    log_emissions = F.log_softmax(self.emission_matrix, dim=1)
    return log_emissions[:, c].transpose(0, 1)


def forward_transition(self, log_alpha):
    """Implements a transition forward step.

    Given the log-alphas from the last timestep, computes the logsumexp over all states of the log transition matrix + the old log-alphas. 

    Arguments
    ---------
    log_alpha : (batch_size, 2) tensor
        Holds log-alphas for all sequences of the last timestep.

    Returns
    -------
    (batch_size, 2) tensor 
        Holds the transition part for the computation of the next log-alphas.
    """
    log_transitions = F.log_softmax(self.transition_matrix, dim=0)
    log_transitions = log_transitions.reshape(1, 2, 2)
    log_alpha = log_alpha.reshape(-1, 1, 2)
    return torch.logsumexp(log_transitions + log_alpha, dim=2)
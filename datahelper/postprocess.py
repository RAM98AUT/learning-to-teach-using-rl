"""
Postprocesses fitted models for using the BKT parameters in reinforcement learning.
"""
import torch
import numpy as np
import pandas as pd
import pickle


class BlockParams:
    """Class to compute, handle and save BKT parameters from fitted HMM models.

    Attributes
    ----------
    block_lengths : list
        List containing the lengths (i.e. number of skills) of the blocks.
    skill_list : list
        List containing all skills of all blocks (ordered as the blocks).
    l0_list : list
        List containing the estimated prior for each skill. 
        Ordering corresponds to skill_list.
    transition_list : list
        List containing the estimated transition probability for each skill.
        Ordering corresponds to skill_list.
    slip_list : list
        List containing the estimated slip probability for each block.
        Ordering corresponds to the input blocks.
    guess_list : list
        List containing the estimated guess probability for each block.
        Ordering corresponds to the input blocks. 
    """

    def __init__(self, models, blocks):
        """Inits the BlockParams class with models and blocks.
        
        Arguments
        ---------
        models : list of lists
            Each sublist contains (a) fitted model(s) for one block.
            len(models) corresponds to the number of blocks.
            In case of multiple models per block, the estimated parameters are averaged.
        blocks : Blocks   
            Instance of the Blocks class holding the block informations.
            Same block ordering as in models is required. 
        """
        self.block_lengths = [len(b) for b in blocks]
        self.skill_list = []
        self.l0_list = []
        self.transition_list = []
        self.slip_list = []
        self.guess_list = []
        
        with torch.no_grad():
            for ms, b in zip(models, blocks):
                block_skills = sorted(b)
                priors = []
                transitions = []
                slips = []
                guesses = []
                for m in ms:
                    # extract parameters
                    priors.append(torch.softmax(m.priors, dim=1).numpy()[:, 1])
                    transitions.append(torch.softmax(m.transition.transition_matrix, dim=1).numpy()[:, 1])
                    slips.append(torch.softmax(m.emission.emission_matrix, dim=1).numpy()[1, 0])
                    guesses.append(torch.softmax(m.emission.emission_matrix, dim=1).numpy()[0, 1])
                # add averages to placeholders
                self.skill_list.extend(block_skills)
                self.l0_list.extend(np.array(priors).mean(axis=0).tolist())
                self.transition_list.extend(np.array(transitions).mean(axis=0).tolist())
                self.slip_list.append(np.mean(slips))
                self.guess_list.append(np.mean(guesses))

    @property
    def dict_(self):
        """Neatly returns the estimated parameters in a dictionary."""
        dict_ = {
            'l0': self.l0_list,
            'transition': self.transition_list,
            'slip': self.slip_list,
            'guess': self.guess_list
        }
        return dict_

    @property
    def df_(self):
        """Neatly returns the estimated parameters in a pd.DataFrame."""
        df_ = pd.DataFrame({
            'skill': self.skill_list,
            'block': np.repeat(range(len(self.block_lengths)), self.block_lengths),
            'l0': self.l0_list,
            'transition': self.transition_list,
            'slip': np.repeat(self.slip_list, self.block_lengths),
            'guess': np.repeat(self.guess_list, self.block_lengths)
        })
        return df_

    def save(self, filename):
        """Saves the instance to the given filename."""
        with open(filename, 'wb') as out:
            pickle.dump(self, out)
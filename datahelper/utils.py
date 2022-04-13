"""
Contains small helper function for dealing with the skillbuilder dataset.
"""
import numpy as np


def get_unique_skills(df, return_counts=False):
    """Finds the unique skills in the skillbuilder dsf based on the skill_name{i} columns.
    
    Arguments
    ---------
    df : pd.DataFrame
        Skillbuilder dataset containing the skill_name{i} columns.
    return_counts : bool, default=False
        cf. return_counts in np.unique.

    Returns
    -------
    unique : ndarray
        The sorted unique skills in the dataset.
    unique_counts : ndarray
        The counts for the unique values, only returned if return_counts is True.
    """
    skill_cols = [col for col in df.columns if col.startswith('skill_name')]
    skills = df[skill_cols].values.reshape(-1,)
    skills = skills[skills != np.array(None)]
    skills = skills[skills != 'None']
    return np.unique(skills, return_counts=return_counts)

def flatten(list_of_lists):
    """Convenience for flattening a list of lists."""
    return [x for sublist in list_of_lists for x in sublist]
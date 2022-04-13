"""
Contains a class with a callable to load the relevant part of the skillbuilder dataset.
"""
import pandas as pd
from .preprocess import update_skill_cols


class SkillbuilderImporter:
    """Class returning a callable for import the skillbuilder dataset.

    The __call__ method loads the columns order_id, user_id, correct, skill_id and
    additional columns provided as argument from some path.

    Arguments
    ---------
    path : str
        Path to the skillbuilder data file.
    additional_cols : list (default=[])
        List of strings containing additional columns to load.

    Returns
    -------
    df : pd.DataFrame
        DataFrame sorted by order_id including skill_name{i} columns (see: update_skill_cols in datahelper.preprocess).
    """  
    
    def __call__(self, path, additional_cols=[]):
        # read specified columns
        usable_cols = ["order_id", "user_id", "correct", "skill_id"] + additional_cols
        df = pd.read_csv(path, usecols=usable_cols, dtype={"skill_id": str})
        # make correct and order_id numeric and sort by order_id
        df['correct'] = df['correct'].astype('int')
        df['order_id'] = df['order_id'].astype('int')
        df = df.sort_values("order_id")
        # create skill_name{i} columns
        df = update_skill_cols(df)
        return df
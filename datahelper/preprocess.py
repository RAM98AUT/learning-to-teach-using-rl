"""
Contains functions to clean and prepare the skillbuilder dataset for fitting.
"""
import pandas as pd
import networkx as nx

from .graphcut import get_skill_graph, Blocks, split_blocks
from .utils import get_unique_skills


def update_skill_cols(df, drop=False):
    """Updates the skill_name{i} columns based on the skill_id column.
    
    Useful whenever the skill_id column has been altered to make the same changes to the skill_name{i} columns.

    Arguments
    ---------
    df : pd.DataFrame
        The skillbuilder dataset.
    drop : bool (default=False)
        Whether or not redundant skill_name{i} columns should be dropped in case the maximum number of skills per exercise has decreased.

    Returns
    -------
    df : pd.DataFrame
        The updated skillbuilder dataset.
    """
    if drop:
        old_skill_cols = [col for col in df.columns if col.startswith('skill_name')]
        df = df.drop(columns=old_skill_cols)
    new_data = df['skill_id'].str.split('_', 0, expand=True)
    new_skill_cols = ['skill_name' + str(i) for i in range(new_data.shape[1])]
    df[new_skill_cols] = new_data
    return df


def remove_null(df):
    """Removes rows where the skill_id column is null."""
    return df[df['skill_id'].notnull()]


def remove_rare_skills(df, threshold=30):
    """Removes skills from skillbuilder with too little occurrences.

    Rare skills are totally removed, but all rows with at least one non-rare skill are retained.
    
    Arguments
    ---------
    df : pd.DataFrame
        The skillbuilder dataset.
    threshold : int (default=30)
        The minimum number of occurrences a skill must have to be retained.

    Returns
    -------
    df : pd.DataFrame
        The updated skillbuilder dataset.
    """
    # find skills with less than threshold occurences
    skills, counts = get_unique_skills(df, return_counts=True)
    skill_counts = pd.DataFrame({'skill': skills, 'count': counts})
    rare_skills = skill_counts[skill_counts['count'] < threshold]['skill'].values
    # replace rare skills by '' in skill_name columns
    skill_cols = ['skill_name' + str(i) for i in range(4)]
    df[skill_cols] = df[skill_cols].replace(list(rare_skills), '')
    df[skill_cols] = df[skill_cols].fillna('')
    # update skill_id column
    df['skill_id'] = ['_'.join(filter(None, x)) for x in zip(df['skill_name0'], df['skill_name1'], df['skill_name2'], df['skill_name3'])]
    # remove obsolete rows
    df = df[df['skill_id'] != '']
    # update (=format) skill_name columns
    df = update_skill_cols(df).astype(str)
    # make numeric columns integer again
    df['correct'] = df['correct'].astype('int')
    df['order_id'] = df['order_id'].astype('int')
    return df


def clean_skill_id(x, keep_skills):
    """Cleans a single skill_id entry keeping only provided skills.
    
    To be used within a call to apply.

    Arguments
    ---------
    x : str
        One skill_id entry.
    keep_skills : set
        Set of skills to keep in x.

    Returns
    -------
    x_new : str
        Cleaned skill_id entry.
    """
    skills = x.split('_')
    skills = keep_skills.intersection(skills)
    return '_'.join(skills)


def blockify(df, blocks):
    """Splits the skillbuilder df into a list of dataframes.
    
    There will be one dataframe for every block.
    Rows concerning multiple blocks will be part of all relevant block dataframes. 

    Arguments
    ---------
    df : pd.DataFrame
        The skillbuilder dataset.
    blocks : Blocks
        An instance of the Blocks class containing the blocks for which to create dataframes.

    Returns
    -------
    block_dfs : list
        A list of dataframes. One dataframe for every block in blocks.
    """
    skill_cols = ['skill_name' + str(i) for i in range(4)]
    # map skill_name columns to block-ids
    skill_block_mapping = {s:i for (i, b) in enumerate(blocks.blocks_) for s in b}
    for col in skill_cols:
        df[col] = df[col].map(skill_block_mapping)  
    # create one dataframe per block
    block_dfs = []
    for block_id in range(len(blocks)):
        # filter relevant rows
        keep_rows = (df[skill_cols] == block_id).any(axis=1)
        block_df = df[keep_rows].reset_index(drop=True)
        # only keep skills of current block
        block_skills = blocks[block_id]
        block_df['skill_id'] = block_df['skill_id'].apply(clean_skill_id, keep_skills=block_skills)
        # fix the skill_name columns
        block_df = update_skill_cols(block_df, drop=True)
        # add to df list
        block_dfs.append(block_df)
    return block_dfs
    

def prepare_fitting(block_df, block):
    """Prepares skill_name columns for fitting.
    
    Instead of skill_name0, skill_name1, ... there will be columns skill_name105, skill_name76, ...
    for a block containing skills 105 and 76 for instance.
    The columns will be a binary encoding in accordance with the format required by the BlockBKTDataset class.

    Arguments
    ---------
    block_df : pd.DataFrame
        A subset of the skillbuilder dataset corresponding to one block.
    block : set
        Set (or list) containing the skills of the considered block.

    Returns
    -------
    block_df : pd.DataFrame
        Updated block dataset.
    """
    # drop the old skill_name columns
    old_skill_cols = [col for col in block_df.columns if col.startswith('skill_name')]
    block_df = block_df.drop(columns=old_skill_cols)
    # create the new skill_name columns
    skills = sorted(block)
    for i in range(len(skills)):
        block_df['skill_name' + str(skills[i])] = block_df['skill_id'].str.split('_').apply(lambda x: skills[i] in x).astype('int') 
    # sort the dataset by order_id
    block_df = block_df.sort_values('order_id')
    return block_df


def preprocess(df, min_exercises=30, max_block_size=5, random_state=None, assign_labels='kmeans'):
    """Sequentially performs all preprocessing steps on the skillbuilder data.
    
    Takes care of
        - removing columns with missing values 
        - removing skills with too little occurrences
        - splitting the skill set into non-overlapping blocks
        - formatting the skill_name columns for fitting

    Arguments
    ---------
    df : pd.DataFrame
        The skillbuilder dataset.
    min_exercises : int (default=30)
        The minimum number of occurrences for a skill to be retained.
    max_block_size : int (default=5)
        The maximum number of skills a block may comprise.
    random_state : int (default=None)
        The random state for the block clustering (kmeans initialization).
    assign_labels : str (default='kmeans')
        The strategy for clustering the embeddingin spectral clustering, one of {'kmeans', 'discretize'}. 

    Returns
    -------
    block_dfs : list
        A list of dataframes, one dataframe for each block.
    blocks : Blocks
        An instance of the Blocks class to more easily keep track of the blocks.
    """
    df = remove_null(df)
    df = remove_rare_skills(df, threshold=min_exercises)
    skill_graph = get_skill_graph(df)
    blocks = list(nx.connected_components(skill_graph))
    blocks = Blocks(blocks)
    split_blocks(blocks, skill_graph, max_block_size, random_state, assign_labels)
    block_dfs = blockify(df, blocks)
    block_dfs = [prepare_fitting(bdf, blk) for (bdf, blk) in zip(block_dfs, blocks)]
    return block_dfs, blocks


# FUNCTIONS FOR SKILLBUILDER EDA NOTEBOOK
def get_skill_mapping(df, max_num_skills, sortkey=lambda x: int(x)):
    skill_cols = ['skill_name' + str(i) for i in range(max_num_skills)]
    unique_skills = pd.unique(df[skill_cols].values.ravel())
    unique_skills = unique_skills[unique_skills != 'None']
    unique_skills = sorted(unique_skills, key=sortkey)
    return {skill:id for (id, skill) in enumerate(unique_skills)}


def apply_skill_mapping(df, max_num_skills, mapping):
    for i in range(max_num_skills):
        col = 'skill_name' + str(i)
        df[col] = df[col].map(mapping)
    return df
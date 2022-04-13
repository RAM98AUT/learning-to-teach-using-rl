"""
Contains classes and functions to use spectral clustering for splitting co-occuring skills into blocks of skills with a maximum block size.
"""
import networkx as nx
import pandas as pd
import numpy as np

from sklearn.cluster import SpectralClustering
from itertools import combinations
from collections import Counter
from math import ceil

from .utils import get_unique_skills, flatten


def get_skill_graph(df):
    """Creates a graph with skills as nodes and edges for co-occurring skills.

    The edge weights are determined by how often a skill pair co-occurs in df.

    Arguments
    ---------
    df : pd.DataFrame
        Skillbuilder dataframe with skill_id and skill_name{i} columns.

    Returns
    -------
    G : nx.Graph
        Graph with all unique skills as nodes and edges for co-occurrences.
    """
    G = nx.Graph()
    # add nodes
    node_list = get_unique_skills(df).tolist()
    G.add_nodes_from(node_list)
    # add edges (with weights)
    skill_pairs = map(lambda string: combinations(sorted(string.split('_')), r=2), df['skill_id'])
    skill_pairs = [list(x) for x in skill_pairs]
    skill_pairs = [s for s in skill_pairs if len(s)>0]
    skill_pairs = flatten(skill_pairs)
    skill_pairs = Counter(skill_pairs)
    edge_list = [(key[0], key[1], {'weight': val}) for (key, val) in skill_pairs.items()]
    G.add_edges_from(edge_list)
    return G


def draw_graph(G):
    """Draws a graph including node labels and edge weights.

    Arguments
    ---------
    G : nx.Graph
        Graph where the edges have an attribute 'weight'.    
    """
    nx.draw(G, labels={n:n for n in G.nodes()})
    nx.draw_networkx_edge_labels(G, pos=nx.spring_layout(G), edge_labels={e:G.edges[e]['weight'] for e in G.edges()})


def split_blocks(blocks, skill_graph, threshold=5, random_state=2022, assign_labels='kmeans'):
    """Splits blocks until all blocks are no larger than a max_skills threshold.

    Does not return but overwrites the blocks instance as a side-effect.

    Arguments
    ---------
    blocks : Blocks
        An instance of the Blocks class containing the initial blocks.
    skill_graph : nx.Graph
        Graph with all unique skills as nodes and edges for co-occurrences as returned by get_skill_graph.
    threshold : int (default=5)
        The maximum number of skills that a block can have in the end.
    random_state : int (default=2022)
        Random state for spectral clustering. Relevant for kmeans initialization.
    assign_labels : str (default='kmeans')
        Strategy for clustering the embedding, one of {'kmeans', 'discretize'}.
    """
    while blocks.max_length_ > threshold:
        component = skill_graph.subgraph(blocks.longest_block_)
        n_clusters = ceil(component.number_of_nodes() / threshold)
        clustering = BlockClustering(component, n_clusters)
        clustering.fit(random_state, assign_labels)
        new_blocks = clustering.blocks_
        blocks.update(new_blocks)


class Blocks:
    """A class representing and managing the blocks for the skillbuilder dataset.

    Attributes
    ----------
    blocks : list of sets
        Each set is a block, i.e. a set of co-occurring skills.
    """

    def __init__(self, blocks):
        """Inits an instance with a list of sets (=blocks)
        
        Arguments
        ---------
        blocks : list of sets
            Each set is a block, i.e. a set of skills co-occurring. 
        """
        self.blocks = blocks

    def update(self, new_blocks):
        """Updates the blocks by replacing an old (large) block by new smaller subblocks.

        The new_blocks are a splitted version where their union is exactly equal to one of the current blocks.
        The current block is removed and the new_blocks are added instead.
        This function takes turns with graph-cutting that always splits a large block into multiple smaller subblocks.
        
        Arguments
        ---------
        new_blocks : list of sets
            The union of the sets is equal to one of the current blocks.
        """
        old_block_index = self.blocks.index(set().union(*new_blocks))
        del self.blocks[old_block_index]
        self.blocks.extend(new_blocks)

    def __len__(self):
        """Returns the number of blocks."""
        return len(self.blocks)

    def __getitem__(self, idx):
        """Returns one of the blocks."""
        return self.blocks[idx]

    @property
    def blocks_(self):
        """Returns the blocks attribute."""
        return self.blocks

    @property 
    def max_length_(self):
        """Returns the length of the longest block."""
        return max(len(b) for b in self.blocks)

    @property
    def longest_block_(self):
        """Returns the longest block."""
        block_lengths = [len(b) for b in self.blocks]
        idx = block_lengths.index(max(block_lengths))
        return self.blocks[idx]


class BlockClustering:
    """Spectral clustering class to split a too large block.
    
    Attributes
    ----------
    nodes : (n_skills,) ndarray
        Contains all unique skills of the block.
    affinity_matrix : (n_skills, n_skills) ndarray
        Adjacency matrix of the block (given by the edge weights) to be used in the splits.
    n_clusters : int
        Number of subblocks to end up with.
    blocks_ : list of sets or None
        The list of blocks (=sets) resulting from spectral clustering.
    """

    def __init__(self, component, n_clusters):
        """Inits the class.
        
        Arguments
        ---------
        component : nx.Graph
            The subgraph of the skill_graph corresponding to the block to be splitted. 
            Has to be a connected component.
        n_clusters : int
            Number of subblocks (=clusters) to end up with. 
        """
        self.nodes = np.array(component.nodes())
        self.affinity_matrix = nx.to_numpy_array(component)
        self.n_clusters = n_clusters
        self.blocks_ = None

    def fit(self, random_state=None, assign_labels='kmeans'):
        """Fits a SpectralClustering model to split the block.
        
        Alters the blocks_ attribute but does not return any output.

        Arguments
        ---------
        random_state : int, default: None
            Random state for spectral clustering. Relevant for kmeans initialization.
        assign_labels : str, default: 'kmeans'
            Strategy for clustering the embedding, one of {'kmeans', 'discretize'}.
        """
        model = SpectralClustering(n_clusters=self.n_clusters, affinity='precomputed', assign_labels=assign_labels, random_state=random_state)
        model.fit(self.affinity_matrix)
        self._create_blocks(model.labels_)

    def _create_blocks(self, labels):
        """Updates the blocks_ attribute after fitting SpectralClustering."""
        self.blocks_ = []
        for i in range(self.n_clusters):
            indices = np.argwhere(labels==i).ravel()
            block = set(self.nodes[indices])
            self.blocks_.append(block)

    @property
    def affinity_matrix_(self):
        """Returns the affinity_matrix attribute as pd.DataFrame."""
        return pd.DataFrame(self.affinity_matrix, columns=self.nodes, index=self.nodes)
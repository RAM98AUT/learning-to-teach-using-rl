"""
Datahelper
==========

Data cleaning, pre- and postprocessing package for the skillbuilder dataset.
The skillbuilder.ipynb in the notebooks folder contains a step-by-step tutorial on how to use this package.

Contains functions and classes for ...
    ... importing the skillbuilder dataset.
    ... cleaning the skillbuilder dataset. (e.g. missing values, rare skills, etc.)
    ... splitting blocks of co-occuring skills to a maximum block size with spectral clustering.
    ... preparing and encoding the skillbuilder dataset for fitting (with torchbkt).
    ... postprocessing fitted models to extract BKT parameters for reinforcement learning.
"""

from .importer import SkillbuilderImporter
from .preprocess import remove_null, remove_rare_skills, get_skill_mapping, apply_skill_mapping
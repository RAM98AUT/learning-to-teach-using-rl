"""
Performs cross-validation with the blockbkt model on the skillbuilder dataset.
Useful to find reasonable hyperparameters (lr, regularization) and find out about the variability of the estimates.
"""
import torch
import pickle

from datahelper.importer import SkillbuilderImporter
from datahelper.preprocess import preprocess
from torchbkt.blockbkt.trainer import BlocksTrainer
from datahelper.postprocess import BlockParams


# importing/exporting parameters
path = 'data/skill_builder_data_corrected_collapsed.csv'
blocks_file = 'output/torchbkt/blocksdfs_blocks_opt.pkl'
blocks_trainer_file = 'output/torchbkt/blocks_trainer_opt.pkl'
bkt_params_file = 'output/torchbkt/bkt_params_opt.pkl'

# preprocessing parameters
min_exercises = 30
max_block_size = 5
random_state = 2022
assign_labels = 'kmeans'

# fitting parameters
newfit = True

lr = 0.02
max_batch_size = 8
min_epochs = 12
min_steps = 120
lr_steps = 2
gamma = 0.1
delta = 0.01
omicron = 0.001
weighted = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

n_splits = 5
verbose = True


if __name__ == '__main__':
    #### 1. LOADING AND PREPROCESSING
    if newfit:
        importer = SkillbuilderImporter()
        data = importer(path)
        block_dfs, blocks = preprocess(data, min_exercises, max_block_size, random_state, assign_labels)
        with open(blocks_file, 'wb') as out:
            pickle.dump((block_dfs, blocks), out)
    else:
        with open(blocks_file, 'rb') as out:
            block_dfs, blocks = pickle.load(out)

    #### 2. FITTING
    if newfit:
        blocks_trainer = BlocksTrainer(lr, max_batch_size, min_epochs, min_steps, lr_steps, gamma, delta, omicron, weighted, device)
        blocks_trainer.cross_validate(blocks, block_dfs, n_splits, verbose)
        with open(blocks_trainer_file, 'wb') as out:
            pickle.dump(blocks_trainer, out)
    else:
        with open(blocks_trainer_file, 'rb') as out: 
            blocks_trainer = pickle.load(out)

    #### 3. POSTPROCESSNG
    block_params = BlockParams(blocks_trainer.models, blocks)
    bkt_params = block_params.dict_
    with open(bkt_params_file, 'wb') as out:
        pickle.dump(bkt_params, out)
"""
Main fitting file, where a blockbkt model is fitted to every block in the skillbuilder dataset after preprocessing.
This is for the final fit on all data without (cross-)validation.
"""
import torch
import pickle

from datahelper.importer import SkillbuilderImporter
from datahelper.preprocess import preprocess
from torchbkt.blockbkt.trainer import BlocksTrainer
from datahelper.postprocess import BlockParams


# importing/exporting parameters
path = 'data/skill_builder_data_corrected_collapsed.csv'
blocks_file = 'output/torchbkt/blocksdfs_blocks_final_fit.pkl'
blocks_trainer_file = 'output/torchbkt/blocks_trainer_final_fit.pkl'
bkt_params_file = 'output/torchbkt/bkt_params_final_fit.pkl'
block_params_file = 'output/torchbkt/block_params_final_fit.pkl'

# preprocessing parameters
min_exercises = 30
max_block_size = 5
random_state = 2022
assign_labels = 'kmeans'

# fitting parameters
newfit = True

lr = 0.02
max_batch_size = 8
min_epochs = 15
min_steps = 150
lr_steps = 2
gamma = 0.1
delta = 0.01
omicron = 0.001
weighted = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        blocks_trainer.fit(blocks, block_dfs, verbose)
        with open(blocks_trainer_file, 'wb') as out:
            pickle.dump(blocks_trainer, out)
    else:
        with open(blocks_trainer_file, 'rb') as out: 
            blocks_trainer = pickle.load(out)

    #### 3. POSTPROCESSNG
    block_params = BlockParams(blocks_trainer.models, blocks)
    bkt_params = block_params.dict_
    with open(block_params_file, 'wb') as out:
        pickle.dump(block_params, out)
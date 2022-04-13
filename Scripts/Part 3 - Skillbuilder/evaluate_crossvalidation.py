"""
Some evaluation/inspection of the cross-validation results.
"""
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler


# importing/exporting parameters
path = 'data/skill_builder_data_corrected_collapsed.csv'
blocks_file = 'output/torchbkt/blocksdfs_blocks_opt.pkl'
blocks_trainer_file = 'output/torchbkt/blocks_trainer_opt.pkl'
bkt_params_file = 'output/torchbkt/bkt_params_opt.pkl'


if __name__ == '__main__':

    #### 0. LOAD INPUT
    # load skillbuilder data
    with open(blocks_file, 'rb') as out:
        block_dfs, blocks = pickle.load(out)

    # load trained blocks trainer
    with open(blocks_trainer_file, 'rb') as out: 
        blocks_trainer = pickle.load(out)


    #### 1. DIFFERENT GROUPS OF BLOCKS (N_SKILLS, N_STUDENTS, N_ROWS) AND HYPERPARAMETERS (LR, N_STEPS, DELTA)
    # extract block characteristics
    n_skills, n_students, n_rows = ([] for i in range(3))
    for df in block_dfs:
        n_skills.append(df.shape[1] - 4)
        n_students.append(df["user_id"].nunique())
        n_rows.append(df.shape[0])

    # create dataframe
    block_characteristics = pd.DataFrame({
        'n_skills': n_skills,
        'n_students': n_students,
        'n_rows': n_rows
    })

    # plot block characteristics
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(n_skills, n_students, n_rows)
    plt.show()

    # preprocess dataframe
    X = block_characteristics.values
    poly = PolynomialFeatures(2)
    X = poly.fit_transform(X)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    # kmeans clustering
    labels = {}
    scores = []
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k).fit(X)
        labels[k] = kmeans.labels_
        scores.append(kmeans.score(X))

    # cluster results (ellbow trick)
    plt.plot(range(2, 10), scores)
    plt.scatter(range(2, 10), scores)
    plt.show()
    block_characteristics['cluster'] = labels[4]

    # plot clustered block characteristics
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for k in range(4):
        subset = block_characteristics['cluster'] == k
        ax.scatter(block_characteristics['n_skills'][subset],
                   block_characteristics['n_students'][subset],
                   block_characteristics['n_rows'][subset])
    plt.show()


    #### 2. LEARNING CURVES
    for (i, (b, df)) in enumerate(zip(blocks, block_dfs)):
        # print block characteristics
        print(f'Cluster ... {block_characteristics["cluster"][i]}')
        print(f'# Rows ... {df.shape[0]}')
        print(f'# Students ... {df["user_id"].nunique()}')
        print(f'# Skills ... {df.shape[1] - 4}')
        # train learning curves on batch level
        for (j, tlc) in enumerate(blocks_trainer.train_learning_curves[i]):
            tlc = np.array(tlc).ravel()
            plt.plot(range(len(tlc)), tlc, label='fold ' + str(j))
            plt.xlabel('batch')
            plt.ylabel('train loss')
            plt.legend()
            plt.show()
        # valid learning curves on epoch level
        for (j, (vlc, tlc)) in enumerate(zip(blocks_trainer.valid_learning_curves[i],
                                            blocks_trainer.train_learning_curves[i])):
            vlc = [x.mean() for x in vlc]
            tlc = [x.mean() for x in tlc]
            plt.plot(range(len(tlc)), tlc, label='train fold ' + str(j))
            plt.plot(range(len(vlc)), vlc, label='valid fold ' + str(j))
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.show()


    #### 2.5 FINAL LOSS VALUES
    final_train_losses = [[x.mean() for x in tlc][-1] for sublist in blocks_trainer.train_learning_curves for tlc in sublist]
    final_valid_losses = [[x.mean() for x in tlc][-1] for sublist in blocks_trainer.valid_learning_curves for tlc in sublist]
    x_axis = range(len(final_train_losses))
    plt.plot(x_axis, sorted(final_train_losses), label='train')
    plt.plot(x_axis, sorted(final_valid_losses), label='valid')
    plt.xlabel('Fold')
    plt.ylabel('Final loss')
    plt.legend()
    plt.grid()
    plt.show()


    #### 3. VARIANCE BETWEEN FOLDS (= WITHIN BLOCKS)
    with torch.no_grad():
        for (i, (ms, df)) in enumerate(zip(blocks_trainer.models, block_dfs)):
            # extract fitted parameters
            priors, transitions, slips, guesses = ([] for i in range(4))
            for m in ms:
                priors.append(torch.softmax(m.priors, dim=1).numpy()[:, 1])
                transitions.append(torch.softmax(m.transition.transition_matrix, dim=1).numpy()[:, 1])
                slips.append(torch.softmax(m.emission.emission_matrix, dim=1).numpy()[1, 0])
                guesses.append(torch.softmax(m.emission.emission_matrix, dim=1).numpy()[0, 1])

            # print block characteristics 
            print(f'Cluster ... {block_characteristics["cluster"][i]}')
            print(f'# Rows ... {df.shape[0]}')
            print(f'# Students ... {df["user_id"].nunique()}')
            print(f'# Skills ... {df.shape[1] - 4}')
            
            # slip and guess
            plt.subplot(3, 1, 1)
            plt.scatter(slips, ['slip'] * len(slips))
            plt.scatter(guesses, ['guess'] * len(guesses))
            plt.xlim(0, 1)

            # transitions
            plt.subplot(3, 1, 2)
            for i in range(len(transitions[0])):
                y = ['transition' + str(i)] * len(transitions)
                x = [t[i] for t in transitions]
                plt.scatter(x, y)
                plt.xlim(0, 1)

            # priors
            plt.subplot(3, 1, 3)
            for i in range(len(priors[0])):
                y = ['prior' + str(i)] * len(priors)
                x = [t[i] for t in priors]
                plt.scatter(x, y)
                plt.xlim(0, 1)
            plt.show()


    #### 4. SEQUENCE LENGTH VS. LIKELIHOOD (WEIGHTING)
    flatten = lambda x: [elem for sublist in x for elem in sublist]
    for (i, (b, df)) in enumerate(zip(blocks, block_dfs)):
        # print block characteristics
        print(f'Cluster ... {block_characteristics["cluster"][i]}')
        print(f'# Rows ... {df.shape[0]}')
        print(f'# Students ... {df["user_id"].nunique()}')
        print(f'# Skills ... {df.shape[1] - 4}')
        # plot
        sequence_lenghts = flatten(blocks_trainer.train_lengths[i])
        log_likelihoods = flatten(blocks_trainer.train_loglikelihoods[i])
        plt.scatter(sequence_lenghts, log_likelihoods, label='train')
        sequence_lenghts = flatten(blocks_trainer.valid_lengths[i])
        log_likelihoods = flatten(blocks_trainer.valid_loglikelihoods[i])
        plt.scatter(sequence_lenghts, log_likelihoods, label='valid')
        plt.legend()
        plt.xlabel('sequence length')
        plt.ylabel('log-likelihood')
        plt.show()
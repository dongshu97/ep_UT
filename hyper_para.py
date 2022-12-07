import os
import optuna
from optuna.trial import TrialState
#Main for the simulation

import argparse
import torch
import torchvision
import torch.optim as optim
import pickle
import numpy as np
import platform
import time
from tqdm import tqdm
from pathlib import Path
import pandas as pd


def argsCreate(epoch, batchSize, lrBias, lrWeights, layersList):

    parser = argparse.ArgumentParser(description='usupervised EP')
    parser.add_argument(
        '--dataset',
        type=str,
        default="mnist",
        help='dataset to be used to train the network : (default = digits, other: mnist)')
    parser.add_argument(
        '--epochs',
        type=int,
        default=epoch,
        metavar='N',
        help='number of epochs to train (default: 50)')
    parser.add_argument(
        '--batchSize',
        type=int,
        default=batchSize,
        help='Batch size (default=1)')
    parser.add_argument(
        '--test_batchSize',
        type=int,
        default=256,
        help='Testing batch size (default=256)')
    parser.add_argument(
        '--layersList',
        nargs='+',
        type=int,
        default=layersList,
        help='List of fully connected layers in the model')
    parser.add_argument(
        '--lrBias',
        type=float,
        default=lrBias,
        help='learning rate for Bias'
    )
    parser.add_argument(
        '--lrWeights',
        type=float,
        default=lrWeights,
        help='learning rate for Weights (default = 0.001)')
    args = parser.parse_args()

    return args


def training(args):
    # Do your training process
    accuracy = 1   # calculate the training error
    return accuracy


def objective(trial):

    '''This is the optimized function which will called by OPTUNA '''

    lrBias = []
    lrBias1 = trial.suggest_float("lrBias1", 1e-5, 1e-1, log=True)
    lrBias.append(lrBias1)
    # lrBias2 = trial.suggest_float("lrBias2", 1e-5, 1e-1, log=True)
    # lrBias.append(lrBias2)

    lrWeights = []
    lrWeights1 = trial.suggest_float("lrWeights1", 1e-5, 1e-1, log=True)
    lrWeights.append(lrWeights1)
    # lrWeights2 = trial.suggest_float("lrWeights2", 1e-5, 1e-1, log=True)
    # lrWeights.append(lrWeights2)

    # define the batch sizes
    batch_size = trial.suggest_int("batchSize", 16, 512)

    # define the layerList
    layersList = [784, 128]

    # define the epoch
    epoch = 100

    # Transfer the args defined by optuna to argParse
    args = argsCreate(epoch, batch_size, lrBias, lrWeights, layersList)

    # Training the model
    for epoch in tqdm(range(args.epoch)):
        # call your training function
        accuracy = training(args)
        trial.report(accuracy, epoch)
        ''' The optuna can do the prune process for you'''
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def createHyperFile():

    # Create the HyperTest folder
    Path('./HyperTest').mkdir(parents=True, exist_ok=True)
    path_hyper = Path(Path.cwd()/'HyperTest')

    filePathList = list(path_hyper.glob('*.csv'))

    # Create the csv file to save the hyper-parameters trained
    if len(filePathList) == 0:
        filename = 'H-1.csv'
        filePath = path_hyper/filename
    else:
        tab = []
        for i in range(len(filePathList)):
            tab.append(int(filePathList[i].stem.split('-')[-1]))
        filename = 'H-' + str(max(tab)+1) + '.csv'
        filePath = path_hyper/filename


    if Path(filePath).is_file():
        dataframe = pd.read_csv(filePath, sep=',', index_col=0)
    else:
        columns_header = ['lrBias1', 'lrWeights1', 'batchSize', 'accuracy']

        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(filePath)

    return dataframe, filePath, path_hyper


def updataDataframe(filePath, dataframe, lrBias1, lrWeights1, batchSize, epoch, accuracy):

    data = [lrBias1, lrWeights1, batchSize, epoch, accuracy]
    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)
    dataframe = pd.concat([dataframe, new_data], axis=0)
    dataframe.to_csv(filePath)

    return dataframe


if __name__ == '__main__':

    dataframe, filePath, path_hyper = createHyperFile()

    study = optuna.create_study(direction="maximize")

    n_trials = 10
    for _ in range(n_trials):
        trial = study.ask()

        lrBias = []
        lrBias1 = trial.suggest_float("lrBias1", 1e-5, 1e-1, log=True)
        lrBias.append(lrBias1)
        # lrBias2 = trial.suggest_float("lrBias2", 1e-5, 1e-1, log=True)
        # lrBias.append(lrBias2)

        lrWeights = []
        lrWeights1 = trial.suggest_float("lrWeights1", 1e-5, 1e-1, log=True)
        lrWeights.append(lrWeights1)
        # lrWeights2 = trial.suggest_float("lrWeights2", 1e-5, 1e-1, log=True)
        # lrWeights.append(lrWeights2)

        # define the batch sizes
        batch_size = trial.suggest_int("batchSize", 16, 512)

        # define the layerList
        layersList = [784, 128]

        # define the epoch
        epochs = 100

        # Transfer the args defined by optuna to argParse
        args = argsCreate(epochs, batch_size, lrBias, lrWeights, layersList)

        # Initialize the pruned trail
        pruned_epoch = "NON"
        # Training the model
        for epoch in tqdm(range(args.epoch)):
            # call your training function
            accuracy = training(args)
            trial.report(accuracy, epoch)
            ''' The optuna can do the prune process for you'''
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                pruned_trial = True
                pruned_epoch = epoch
                break
                #raise optuna.exceptions.TrialPruned()

        if pruned_trial:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)  # tell the pruned state
        else:
            study.tell(trial, accuracy)
    dataframe = updataDataframe(filePath, dataframe, lrBias1, lrWeights1, batch_size, pruned_epoch, accuracy)
    dataframe = updataDataframe(filePath, dataframe, lrBias1, lrWeights1, batch_size, pruned_epoch, accuracy)


        ## The optimazation can simply realized by the following command, but it can not return the middle states
    # study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))










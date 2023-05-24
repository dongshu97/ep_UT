import optuna
import copy as CP
import torch
import numpy as np
import pandas
import torchvision
# import argparse
import json
import os
import logging
import sys
from pathlib import Path
from Data import *
from Network import *
from Tools import *

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

# load the parameters in optuna_config
with open('.'+prefix + 'optuna_config.json') as f:
  pre_config = json.load(f)

# define the activation function
if pre_config['activation_function'] == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))

    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))

elif pre_config['activation_function'] == 'hardsigm':
    def rho(x):
        return x.clamp(min=0).clamp(max = 1)

    def rhop(x):
        return (x >= 0) & (x <= 1)

elif pre_config['activation_function'] == 'half_hardsigm':
    def rho(x):
        return (1 + F.hardtanh(x - 1))*0.5
    def rhop(x):
        return ((x >= 0) & (x <= 2))*0.5

elif pre_config['activation_function'] == 'tanh':
    def rho(x):
        return torch.tanh(x)

    def rhop(x):
        return 1 - torch.tanh(x)**2


# define the dataset
def returnMNIST(jparams):
    print('We use the MNIST Dataset')
    # Define the Transform
    # !! Attention it depends on whether use the convolutional layers
    if jparams['convNet']:
        transforms = [torchvision.transforms.ToTensor()]
    else:
        transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

    # Down load the MNIST dataset
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms),
                                               target_transform=ReshapeTransformTarget(10))

    validation_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    x = train_set.data
    y = train_set.targets

    class_set = splitClass(x, y, 0.02, seed=jparams['class_seed'],
                           transform=torchvision.transforms.Compose(transforms))

    layer_set = splitClass(x, y, 0.02, seed=jparams['class_seed'],
                           transform=torchvision.transforms.Compose(transforms),
                           target_transform=ReshapeTransformTarget(10))

    # load the datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=jparams['test_batchSize'], shuffle=True)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=1000, shuffle=True)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=1000, shuffle=True)
    #
    if jparams['littleData']:
        targets = train_set.targets
        semi_seed = jparams['semi_seed']
        supervised_dataset, unsupervised_dataset = Semisupervised_dataset(train_set.data, targets,
                                                                          jparams['fcLayers'][0], jparams['n_class'],
                                                                          jparams['trainLabel_number'],transform=torchvision.transforms.Compose(transforms),
                                                                          seed=semi_seed)
        supervised_loader = torch.utils.data.DataLoader(supervised_dataset, batch_size=jparams['pre_batchSize'],
                                                        shuffle=True)
        unsupervised_loader = torch.utils.data.DataLoader(unsupervised_dataset, batch_size=jparams['batchSize'],
                                                          shuffle=True)
        return train_loader, test_loader, class_loader, layer_loader, supervised_loader, unsupervised_loader
    else:
        return train_loader, test_loader, class_loader, layer_loader


def returnYinYang(batchSize, batchSizeTest=128):
    print('We use the YinYang dataset')

    train_set = YinYangDataset(size=5000, seed=42, target_transform=ReshapeTransformTarget(3))
    validation_set = YinYangDataset(size=1000, seed=41)  # used for the hyperparameter research
    class_set = YinYangDataset(size=1000, seed=42, sub_class=True)
    layer_set = YinYangDataset(size=1000, seed=42, target_transform=ReshapeTransformTarget(3), sub_class=True)

    # test_set = YinYangDataset(size=1000, seed=40)
    # classTest_set = YinYangDataset(size=1000, seed=40, sub_class=True)

    # seperate the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batchSizeTest, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSizeTest, shuffle=False)
    # classTest_loader = torch.utils.data.DataLoader(classTest_set, batch_size=100, shuffle=False)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=100, shuffle=True)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=100, shuffle=True)

    return train_loader, validation_loader, class_loader, layer_loader


def jparamsCreate(pre_config, trial):

    jparams = CP.deepcopy(pre_config)

    if jparams["dataset"] == 'mnist':
        #jparams["class_seed"] = trial.suggest_int("class_seed", 0, 42)
        jparams["class_seed"] = 34
        if jparams["littleData"]:
            # jparams["semi_seed"] = trial.suggest_int("semi_seed", 0, 42)
            jparams["semi_seed"] = 13
            jparams["pre_batchSize"] = trial.suggest_int("pre_batchSize", 10, min(jparams["trainLabel_number"], 512))

    if jparams["action"] == 'unsupervised_ep':
        if jparams['Homeo_mode'] == 'batch':
            jparams["batchSize"] = trial.suggest_int("batchSize", 10, 256)
            jparams["eta"] = None
        else:
            jparams["batchSize"] = 1,
            jparams["eta"] = trial.suggest_float("eta", 0.01, 1, log=True)

        jparams["gamma"] = trial.suggest_float("gamma", 0.01, 1, log=True)
        jparams["nudge_N"] = trial.suggest_int("nudge_N", 1, jparams["nudge_max"])

        jparams["beta"] = trial.suggest_float("beta", 0.05, 0.8)
        lr = []
        for i in range(jparams["numLayers"]-1):
            lr_i = trial.suggest_float("lr"+str(i), 1e-3, 1, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams['lr'].reverse()

        jparams["Optimizer"] = 'SGD'
        #jparams["Optimizer"] = trial.suggest_categorical("Optimizer", ['SGD', 'Adam'])

        # if jparams["numLayers"] <= 2:
        #     jparams["errorEstimate"] = 'one-sided'
        # else:
        #     jparams["errorEstimate"] = trial.suggest_categorical("errorEstimate", ['one-sided', 'symmetric'])

        # if jparams["Dropout"]:
        #     dropProb = []
        #     dropProb.append(0.2)
        #     for i in range(1, jparams["numLayers"]):
        #         if jparams["fcLayers"][-1] == jparams["n_class"] and i == jparams["numLayers"]-1:
        #             drop_i = 0
        #         else:
        #             drop_i = trial.suggest_float("drop" + str(i), 0.01, 1, log=True)
        #         dropProb.append(drop_i)
        #     jparams["dropProb"] = dropProb.copy()
        #     jparams["dropProb"].reverse()
        jparams["dropProb"] = [0.2, 0.3]
        jparams["dropProb"].reverse()

        if jparams["Prune"] == "Initiation":
            pruneAmount = []
            for i in range(jparams["numLayers"] - 1):
                prune_i = trial.suggest_float("prune" + str(i), 0.01, 1, log=True)
                # to verify whether we need to change the name of drop_i
                pruneAmount.append(prune_i)
            jparams["pruneAmount"] = pruneAmount.copy()
            jparams["pruneAmount"].reverse()

    elif jparams["action"] == 'supervised_ep':
        if jparams["littleData"]:
            jparams["batchSize"] = 128
        else:
            jparams["batchSize"] = trial.suggest_int("batchSize", 10, 512)
        jparams["eta"] = None
        jparams["gamma"] = None
        jparams["nudge_N"] = None
        jparams["beta"] = trial.suggest_float("beta", 0.05, 0.5)
        lr = []
        for i in range(jparams["numLayers"] - 1):
            lr_i = trial.suggest_float("lr" + str(i), 1e-6, 1e-2, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams["lr"].reverse()

        jparams["Optimizer"] = trial.suggest_categorical("Optimizer", ['SGD', 'Adam'])
        if jparams["numLayers"] == 2:
            jparams["errorEstimate"] = 'one-sided'
        else:
            jparams["errorEstimate"] = trial.suggest_categorical("errorEstimate", ['one-sided', 'symmetric'])
            # jparams["lossFunction"] = trial.suggest_categorical("lossFunction", ['MSE', 'Cross-entropy'])

        if jparams["Dropout"]:
            jparams["dropProb"] = [0.2, 0.5, 0]
            jparams["dropProb"].reverse()

    elif jparams["action"] == 'semi-supervised_ep':

        pre_lr = []
        for i in range(jparams["numLayers"] - 1):
            pre_lr_i = trial.suggest_float("lr" + str(i), 1e-5, 1, log=True)
            # to verify whether we need to change the name of lr_i
            pre_lr.append(pre_lr_i)
        jparams["pre_lr"] = pre_lr.copy()
        jparams["pre_lr"].reverse()
        jparams["eta"] = None
        jparams["gamma"] = trial.suggest_float("gamma", 1e-5, 1, log=True)
        jparams["beta"] = trial.suggest_float("beta", 0.01, 1)
        jparams["nudge_N"] = 1
        jparams["Optimizer"] = trial.suggest_categorical("Optimizer", ['SGD', 'Adam'])
        jparams["batchSize"] = trial.suggest_int("batchSize", 10, 512)
        lr = []
        for i in range(jparams["numLayers"]-1):
            lr_i = trial.suggest_float("lr"+str(i), 1e-7, 0.1, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams["lr"].reverse()
        if jparams["numLayers"] == 2:
            jparams["errorEstimate"] = 'one-sided'
        else:
            jparams["errorEstimate"] = trial.suggest_categorical("errorEstimate", ['one-sided', 'symmetric'])
            # jparams["lossFunction"] = trial.suggest_categorical("lossFunction", ['MSE', 'Cross-entropy'])

        if jparams["Dropout"]:
            dropProb = []
            dropProb.append(0.2)
            for i in range(1, jparams["numLayers"]):
                if jparams["fcLayers"][-1] == jparams["n_class"] and i == jparams["numLayers"] - 1:
                    drop_i = 0
                else:
                    drop_i = trial.suggest_float("drop" + str(i), 0.01, 1, log=True)
                dropProb.append(drop_i)
            jparams["dropProb"] = dropProb.copy()
            jparams["dropProb"].reverse()

    elif jparams["action"] == 'class_layer':
        jparams["batchSize"] = 128
        jparams["beta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["errorEstimate"] = 'one-sided'
        jparams["lr"] = [0.01, 0.02]
        jparams["class_activation"] = 'softmax'
        jparams["class_Optimizer"] = trial.suggest_categorical("class_Optimizer", ['Adam', 'SGD'])
        jparams["class_lr"] = trial.suggest_float("class_lr", 1e-5, 0.1, log=True)

    return jparams


def train_validation(jparams, net, trial, validation_loader, optimizer, train_loader=None, class_loader=None, layer_loader=None,
                     class_net=None, supervised_loader=None, unsupervised_loader=None):
    # train the model
    if jparams['action'] == 'supervised_ep':
        if train_loader is not None:
            print("Training the model with supervised ep")
        else:
            raise ValueError("training data is not given ")

        for epoch in tqdm(range(jparams['epochs'])):
            if jparams['lossFunction'] == 'MSE':
                train_error_epoch = train_supervised_ep(net, jparams, train_loader, optimizer, epoch)
            elif jparams['lossFunction'] == 'Cross-entropy':
                train_error_epoch = train_supervised_crossEntropy(net, jparams, train_loader, optimizer, epoch)

            validation_error_epoch = test_supervised_ep(net, jparams, validation_loader)

            # Handle pruning based on the intermediate value.
            trial.report(validation_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return validation_error_epoch

    elif jparams['action'] == 'unsupervised_ep':
        if train_loader is not None and class_loader is not None:
            print("Training the model with unsupervised ep")
        else:
            raise ValueError("training data or class data is not given ")

        for epoch in tqdm(range(jparams['epochs'])):
            # train process
            Xth = train_unsupervised_ep(net, jparams, train_loader, optimizer, epoch)
            # class process
            response, max0_indice = classify(net, jparams, class_loader)
            # test process
            error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, validation_loader, response)

            # Handle pruning based on the intermediate value.
            trial.report(error_av_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return error_av_epoch

    elif jparams['action'] == 'semi-supervised_ep':
        if supervised_loader is not None and unsupervised_loader is not None:
            print("Training the model with semi-supervised ep")
        else:
            raise ValueError("supervised training data or unsupervised training data is not given ")
        for epoch in tqdm(range(jparams["pre_epochs"])):
            if jparams['lossFunction'] == 'MSE':
                train_error_epoch = train_supervised_ep(net, jparams, supervised_loader, optimizer, epoch)
            elif jparams['lossFunction'] == 'Cross-entropy':
                train_error_epoch = train_supervised_crossEntropy(net, jparams, supervised_loader, optimizer, epoch)

            validation_error_epoch = test_supervised_ep(net, jparams, validation_loader)
            # Handle pruning based on the intermediate value.
            trial.report(validation_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        unsupervised_params, unsupervised_optimizer = defineOptimizer(net, jparams['convNet'], jparams['lr'], jparams['Optimizer'])

        for epoch in tqdm(range(jparams["epochs"])):
            # supervised reminder
            if jparams['lossFunction'] == 'MSE':
                pretrain_error_epoch = train_supervised_ep(net, jparams, supervised_loader, optimizer, epoch)
            elif jparams['lossFunction'] == 'Cross-entropy':
                pretrain_error_epoch = train_supervised_crossEntropy(net, jparams, supervised_loader, optimizer,
                                                                     epoch)
            # unsupervised training
            if jparams['lossFunction'] == 'MSE':
                Xth = train_unsupervised_ep(net, jparams, unsupervised_loader, unsupervised_optimizer, epoch)
            elif jparams['lossFunction'] == 'Cross-entropy':
                Xth = train_unsupervised_crossEntropy(net, jparams, unsupervised_loader, unsupervised_optimizer, epoch)
            entire_test_epoch = test_supervised_ep(net, jparams, validation_loader)
            # Handle pruning based on the intermediate value.
            trial.report(entire_test_epoch, epoch+jparams['pre_epochs'])
            if trial.should_prune():
                raise optuna.TrialPruned()

        return entire_test_epoch

    elif jparams['action'] == 'class_layer':
        if class_net is not None and layer_loader is not None:
            print("Training the model with unsupervised ep")
        else:
            raise ValueError("class net or labeled class data is not given ")

        for epoch in tqdm(range(jparams["class_epoch"])):
            # we train the classification layer
            class_train_error_epoch = classify_network(net, class_net, jparams, layer_loader)
            # test error
            final_test_error_epoch, final_loss_epoch = test_unsupervised_ep_layer(net, class_net, jparams, validation_loader)
            trial.report(final_test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return final_test_error_epoch


def objective(trial, pre_config):

    # design the hyperparameters to be optimized
    jparams = jparamsCreate(pre_config, trial)

    jparams['fcLayers'].reverse()  # we put in the other side, output first, input last
    jparams['C_list'].reverse()  # we reverse also the list of channels

    # create the dataset
    if jparams["dataset"] == 'YinYang':
        train_loader, validation_loader, class_loader, layer_loader = \
            returnYinYang(jparams['batchSize'], batchSizeTest=jparams["test_batchSize"])
    elif jparams["dataset"] == 'mnist':
        if jparams['littleData']:
            train_loader, validation_loader, class_loader, layer_loader, \
            supervised_loader, unsupervised_loader = returnMNIST(jparams)
        else:
            train_loader,  validation_loader, class_loader, layer_loader = returnMNIST(jparams)

    # create the model
    if jparams['convNet']:
        net = ConvEP(jparams, rho, rhop)
    else:
        net = torch.jit.script(MlpEP(jparams, rho, rhop))
    # TODO to include the CNN version
    if jparams['pre_epochs'] > 0:
        initial_lr = jparams['pre_lr']
    else:
        initial_lr = jparams['lr']

    # define the optimizer
    net_params, optimizer = defineOptimizer(net, jparams['convNet'], initial_lr, jparams['Optimizer'])

    # load the trained unsupervised network when we train classification layer
    if jparams["action"] == 'class_layer':
        with open(r'D:\Results_data\EP_batchHomeo\784-2000-N7-beta0.31-hardsigm-lr0.0159-batch139-gamma0.2-epoch100\S-15\model_entire.pt','rb') as f:
            loaded_net = torch.jit.load(f)
        net.W = loaded_net.W.copy()
        net.bias = loaded_net.bias.copy()
        net.eval()

        # create the new class_net
        class_net = Classlayer(jparams)

        final_err = train_validation(jparams, net, trial, validation_loader, layer_loader=layer_loader, class_net=class_net)

    elif jparams["action"] == 'unsupervised_ep':
        final_err = train_validation(jparams, net, trial, validation_loader, optimizer, train_loader=train_loader, class_loader=class_loader)
    elif jparams["action"] == 'supervised_ep':
        if jparams['littleData']:
            final_err = train_validation(jparams, net, trial, validation_loader, optimizer, train_loader=supervised_loader)
        else:
            final_err = train_validation(jparams, net, trial, validation_loader, optimizer, train_loader=train_loader)
    elif jparams["action"] == 'semi-supervised_ep':
        final_err = train_validation(jparams, net, trial, validation_loader, optimizer, supervised_loader=supervised_loader, unsupervised_loader=unsupervised_loader)

    del(jparams)
    # record trials
    df = study.trials_dataframe()
    df.to_csv(filePath)

    return final_err

def optuna_createPath():
    '''
    Create path to save data
    '''
    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    BASE_PATH +=  prefix + 'Optuna-0'

    # BASE_PATH += prefix + args.dataset
    #
    # BASE_PATH += prefix + 'method-' + args.method
    #
    # BASE_PATH += prefix + args.action
    #
    # BASE_PATH += prefix + str(len(args.fcLayers)-2) + 'hidden'
    # BASE_PATH += prefix + 'hidNeu' + str(args.layersList[1])
    #
    # BASE_PATH += prefix + 'β-' + str(args.beta)
    # BASE_PATH += prefix + 'dt-' + str(args.dt)
    # BASE_PATH += prefix + 'T-' + str(args.T)
    # BASE_PATH += prefix + 'K-' + str(args.Kmax)
    #
    # BASE_PATH += prefix + 'Clamped-' + str(bool(args.clamped))[0]
    #
    # BASE_PATH += prefix + 'lrW-' + str(args.lrWeights)
    # BASE_PATH += prefix + 'lrB-' + str(args.lrBias)
    #
    # BASE_PATH += prefix + 'BaSize-' + str(args.batchSize)

    BASE_PATH += prefix + datetime.datetime.now().strftime("%Y-%m-%d")

    if not os.path.exists(BASE_PATH):
        os.makedirs(BASE_PATH)
    print("len(BASE_PATH)="+str(len(BASE_PATH)))
    filePath = shutil.copy('plotFunction.py', BASE_PATH)

    files = os.listdir(BASE_PATH)

    if 'plotFunction.py' in files:
        files.pop(files.index('plotFunction.py'))

    if not files:
        BASE_PATH = BASE_PATH + prefix + 'S-1'
    else:
        tab = []
        for names in files:
            tab.append(int(names.split('-')[1]))
        BASE_PATH += prefix + 'S-' + str(max(tab)+1)


    os.mkdir(BASE_PATH)
    name = BASE_PATH.split(prefix)[-1]

    return BASE_PATH, name


if __name__=='__main__':
    # define prefix
    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # define Sampler

    # define Pruner

    # define the dataframe
    BASE_PATH, name = optuna_createPath()

    # save the optuna configuration
    with open(BASE_PATH + prefix + "optuna_config.json", "w") as outfile:
        json.dump(pre_config, outfile)

    # create the filepath for saving the optuna trails
    filePath = BASE_PATH + prefix + "test.csv"
    study_name = str(time.asctime())
    study = optuna.create_study(direction="minimize", study_name=study_name, storage='sqlite:///example.db')

    study.enqueue_trial(
        {
            "batchSize": 64,
            "gamma": 0.8,
            "nudge_N": 1,
            "beta": 0.5,
            "lr0": 0.6,
            # "lr1" : 0.02,
        }
    )

    study.enqueue_trial(
        {
            "batchSize": 128,
            "gamma": 0.25,
            "nudge_N": 1,
            "beta": 0.5,
            "lr0": 0.4,
            #"lr1" : 0.02,
        }
    )

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(lambda trial: objective(trial, pre_config), n_trials=200)

    trails = study.get_trials()
    # TODO record the trials each trails
    # record trials
    df = study.trials_dataframe()
    df.to_csv(filePath)

    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_slice(study)

    #np.savetxt(BASE_PATH + prefix + "test.csv", trails, delimiter=",", fmt='%s')
    #np.savetxt(BASE_PATH+"test.csv", trails, delimiter=",", fmt='%s', header=header)
    # save study and read the parameters in the study









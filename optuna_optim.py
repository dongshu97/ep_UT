import optuna
import copy as CP
import torch
import numpy as np
import pandas
import torchvision
import argparse
import json
import os
import logging
import sys
from pathlib import Path
from Data import *
from actions import *

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'.',
    help='path of json configuration'
)
parser.add_argument(
    '--trained_path',
    type=str,
    # default=r'.',
    default=r'.\pretrain_file',
    help='path of model_dict_state_file'
)

args = parser.parse_args()

# load the parameters in optuna_config
with open(args.json_path +prefix + 'optuna_config.json') as f:
  pre_config = json.load(f)


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

    if jparams["action"] == 'unsupervised_ep':
        # not used parameters
        jparams["eta"] = None
        # unchanged parameters
        # jparams["gamma"] = 0.3
        jparams["beta"] = 0.2
        # jparams["batchSize"] = 16
        jparams["nudge_N"] = 5
        jparams["scheduler"] = 'linear'
        jparams['factor'] = 0.001
        jparams['scheduler_epoch'] = 100

        # test parameters
        jparams["gamma"] = trial.suggest_float("gamma", 0.01, 0.9, log=True)
        jparams["batchSize"] = trial.suggest_categorical("batchSize", [16, 32, 64])
        # jparams["nudge_N"] = trial.suggest_int("nudge_N", 2, jparams["nudge_max"])

        lr = []
        for i in range(jparams["numLayers"]-1):
            lr_i = trial.suggest_float("lr"+str(i), 1e-6, 1, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams['lr'].reverse()

        # jparams['factor'] = trial.suggest_float("factor", 1e-7, 1, log=True)
        # jparams['scheduler_epoch'] = trial.suggest_int("scheduler_epoch", 1, jparams['epochs'])
        # jparams["scheduler"] = trial.suggest_categorical("scheduler",
        #                                                  ['linear', 'step', 'cosine'])
        jparams["exp_factor"] = 0.5
        # jparams["exp_factor"] = trial.suggest_float("pre_gamma", 0.9, 0.999, log=True)

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

        # if jparams["Dropout"]:
        #     jparams["dropProb"] = [0.2, 0.5, 0]
        #     jparams["dropProb"].reverse()

    elif jparams["action"] == 'semi-supervised_ep':
        # unchanged
        jparams["pre_lr"].reverse()
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.1
        jparams["beta"] = 0.45
        jparams["nudge_N"] = 1
        jparams["Optimizer"] = 'Adam'
        jparams["lossFunction"] = 'Cross-entropy'
        # jparams["dropProb"] = [0.4, 0.5, 0]
        # jparams["dropProb"].reverse()
        jparams["batchSize"] = 128
        # test parameters
        lr = []
        for i in range(jparams["numLayers"]-1):
            lr_i = trial.suggest_float("lr"+str(i), 1e-4, 1e-2, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams["lr"].reverse()
        jparams["supervised_start"] = trial.suggest_float("supervised_start", 1e-3, 1e-1, log=True)
        jparams["supervised_end"] = trial.suggest_float("supervised_end", 1e-3, 1e-1, log=True)
        jparams["unsupervised_start"] = trial.suggest_float("unsupervised_start", 1e-4, 1e-1, log=True)
        jparams["unsupervised_end"] = trial.suggest_float("unsupervised_end", 1e-4, 1e-1, log=True)

    elif jparams["action"] == 'pre_train_ep':
        # the non-used parameters
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.1
        jparams["nudge_N"] = 1
        jparams["batchSize"] = 256
        jparams['lr'] = [0.1, 0.1]
        # the unchanged parameters
        jparams["beta"] = 0.45
        jparams["Optimizer"] = 'Adam'
        jparams["lossFunction"] = 'Cross-entropy'
        # jparams["dropProb"] = [0.4, 0.5, 0]
        # jparams["dropProb"].reverse()
        jparams["pre_batchSize"] = 32
        # test parameters
        pre_lr = []
        for i in range(jparams["numLayers"] - 1):
            pre_lr_i = trial.suggest_float("lr" + str(i), 1e-5, 1, log=True)
            pre_lr.append(pre_lr_i)
        jparams["pre_lr"] = pre_lr.copy()
        jparams["pre_lr"].reverse()
        jparams["pre_scheduler"] = trial.suggest_categorical("pre_scheduler",
                                                             ['linear', 'exponential', 'step', 'cosine'])
        jparams["pre_scheduler_epoch"] = trial.suggest_int("pre_scheduler_epoch", 1, jparams['pre_epochs'])
        jparams["pre_factor"] = trial.suggest_float("pre_factor", 1e-5, 1, log=True)
        jparams["pre_exp_factor"] = trial.suggest_float("pre_exp_factor", 0.9, 0.999, log=True)

    elif jparams["action"] == 'train_class_layer':
        # non-used parameters
        jparams["batchSize"] = 256
        jparams["beta"] = 0.5
        jparams["pre_batchSize"] = 32
        jparams["errorEstimate"] = 'one-sided'
        jparams["eta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["lr"] = [6, 3.35, 2.97, 1.83]
        jparams["nudge_N"] = 1
        jparams['Optimizer'] = 'SGD'
        # unchanged parameters
        jparams["class_Optimizer"] = 'Adam'
        # class
        jparams["test_batchSize"] = trial.suggest_categorical("test_batchSize", [128, 256, 512])
        jparams["class_dropProb"] = trial.suggest_float("class_dropProb", 0, 0.3)
        jparams["class_smooth"] = trial.suggest_categorical("class_smooth", [True, False])
        jparams["class_activation"] = trial.suggest_categorical("class_activation", ['sigmoid', 'x', 'hardsigm'])
        jparams["class_lr"] = [trial.suggest_float("class_lr", 1e-4, 1, log=True)]
        jparams["class_scheduler"] = trial.suggest_categorical("class_scheduler",
                                                               ['linear', 'exponential'])
        jparams["class_scheduler_epoch"] = trial.suggest_int("class_scheduler_epoch", 1, jparams['class_epoch'])
        jparams["class_factor"] = trial.suggest_float("class_factor", 1e-4, 1, log=True)
        jparams["class_exp_factor"] = trial.suggest_float("pre_exp_factor", 0.9, 0.999, log=True)

    return jparams


def objective(trial, pre_config):
    # design the hyperparameters to be optimized
    jparams = jparamsCreate(pre_config, trial)

    # create the dataset
    if jparams["dataset"] == 'YinYang':
        train_loader, validation_loader, class_loader, layer_loader = \
            returnYinYang(jparams['batchSize'], batchSizeTest=jparams["test_batchSize"])
    # TODO define mnist and cifar
    elif jparams["dataset"] == 'mnist':
        print('We use the MNIST Dataset')
        (train_loader, test_loader,
         class_loader, layer_loader,
         supervised_loader, unsupervised_loader) = returnMNIST(jparams, validation=True)
    elif jparams["dataset"] == 'cifar10':
        print('We use the CIFAR10 dataset')
        (train_loader, test_loader,
         class_loader, layer_loader,
         supervised_loader, unsupervised_loader) = returnCifar10(jparams, validation=True)

    # reverse the layer
    jparams['fcLayers'].reverse()  # we put in the other side, output first, input last
    jparams['C_list'].reverse()
    jparams['dropProb'].reverse()
    # create the model
    if jparams['convNet']:
        net = ConvEP(jparams, rho, rhop)
    else:
        net = torch.jit.script(MlpEP(jparams, rho, rhop))
    # TODO to include the CNN version

    # load the trained unsupervised network when we train classification layer
    # if jparams["action"] == 'class_layer':
    #     with open(r'D:\Results_data\EP_batchHomeo\784-2000-N7-beta0.31-hardsigm-lr0.0159-batch139-gamma0.2-epoch100\S-15\model_entire.pt','rb') as f:
    #         loaded_net = torch.jit.load(f)
    #     net.W = loaded_net.W.copy()
    #     net.bias = loaded_net.bias.copy()
    #     net.eval()
    #
    #     # create the new class_net
    #     class_net = Classlayer(jparams)
    #     final_err = train_validation(jparams, net, trial, validation_loader, layer_loader=layer_loader, class_net=class_net)

    if jparams["action"] == 'unsupervised_ep':
        print("Training the model with unsupervised ep")
        final_err = unsupervised_ep(net, jparams, train_loader, class_loader, test_loader, layer_loader, trial=trial)
    elif jparams["action"] == 'supervised_ep':
        print("Training the model with supervised ep")
        final_err = supervised_ep(net, jparams, train_loader, test_loader, trial=trial)
    elif jparams["action"] == 'semi-supervised_ep':
        trained_path = args.trained_path + prefix + 'model_pre_supervised_entire.pt'
        final_err = semi_supervised_ep(net, jparams, supervised_loader, unsupervised_loader, test_loader,
                           trained_path=trained_path, trial=trial)
    elif jparams["action"] == 'pre_train_ep':
        print("Training the model with little dataset with supervised ep")
        final_err = pre_supervised_ep(net, jparams, supervised_loader, test_loader, trial=trial)

    elif jparams["action"] == 'train_class_layer':
        print("Train the supplementary class layer for unsupervised learning")
        trained_path = args.trained_path + prefix + 'model_entire.pt'
        final_err = train_class_layer(net, jparams, layer_loader, test_loader, trained_path=trained_path, trial=trial)


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
    # search_space = {
    #     'factor': [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
    #     'scheduler_epoch': [10, 15, 20, 25, 30, 35, 40, 45, 50],
    #     'lr0': [0.005, 0.01, 0.015, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7],
    #     'nudge_N': [4, 5, 6, 7, 8, 9, 10],
    #     'gamma': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # }
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.RandomSampler(),
                                pruner=optuna.pruners.PercentilePruner(30, n_startup_trials=3, n_warmup_steps=5),
                                study_name=study_name, storage='sqlite:///optuna_ep_Conv.db')

    # study = optuna.create_study(direction="minimize", sampler=optuna.samplers.GridSampler(search_space),
    #                             pruner=optuna.pruners.PercentilePruner(45, n_startup_trials=3, n_warmup_steps=10),
    #                             study_name=study_name, storage='sqlite:///optuna_ep.db')


    # study = optuna.create_study(direction="minimize", study_name=study_name, storage='sqlite:///eqprop_example.db')

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study.optimize(lambda trial: objective(trial, pre_config), n_trials=400)

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









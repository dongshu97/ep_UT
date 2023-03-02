import optuna
import copy as CP
import torch
import numpy as np
import pandas
import torchvision
# import argparse
import json
from pathlib import Path
from Data import *
from Network_optuna import *
from Tools_optuna import *

# load the parameters in optuna_config
with open('.\optuna_config.json') as f:
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

elif pre_config['activation_function'] == 'tanh':
    def rho(x):
        return torch.tanh(x)

    def rhop(x):
        return 1 - torch.tanh(x)**2


# define the dataset
def returnMNIST(class_seed, batchSize, batchSizeTest=256):
    # we will not use the validation set for MNIST in the hyperparameter research
    # we take the influence of class seed into consideration
    # TODO split the dataset by the train_test_split
    # define the optimization dataset
    print('We use the MNIST Dataset')
    # Define the Transform
    # !! Attention it depends on whether use the convolutional layers

    transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

    # Down load the MNIST dataset
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms),
                                               target_transform=ReshapeTransformTarget(10))

    validation_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    # rest_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
    #                                       transform=torchvision.transforms.Compose(transforms))

    class_set = ClassDataset(root='./MNIST_class_seed', test_set=validation_set, seed=class_seed,
                             transform=torchvision.transforms.Compose(transforms))
    layer_set = ClassDataset(root='./MNIST_class_seed', test_set=validation_set, seed=class_seed,
                             transform=torchvision.transforms.Compose(transforms),
                             target_transform=ReshapeTransformTarget(10))

    # load the datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batchSizeTest, shuffle=True)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=1000, shuffle=True)
    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=1000, shuffle=True)

    return train_loader, test_loader, class_loader, layer_loader


def returnYinYang(batchSize, batchSizeTest=128):
    print('We use the YinYang dataset')

    train_set = YinYangDataset(size=5000, seed=42, target_transform=ReshapeTransformTarget(3))
    validation_set = YinYangDataset(size=1000, seed=41)  # used for the hyperparameter research
    class_set = YinYangDataset(size=1000, seed=41, sub_class=True)
    layer_set = YinYangDataset(size=1000, seed=41, target_transform=ReshapeTransformTarget(3), sub_class=True)

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
        jparams["class_seed"] = trial.suggest_int("class_seed", 1, 9)

    if jparams["action"]=='unsupervised_ep':
        if jparams['Homeo_mode']=='batch':
            jparams["batchSize"] = trial.suggest_int("batchSize", 10, 256)
            jparams["eta"] = None
        else:
            jparams["batchSize"] = 1,
            jparams["eta"] = trial.suggest_float("eta", 0.001, 1, log=True)

        jparams["gamma"] = trial.suggest_float("gamma", 0.001, 1, log=True)
        jparams["nudge_N"] = trial.suggest_int("nudge_N", 1, jparams["nudge_max"])

        jparams["beta"] = trial.suggest_float("beta", 0.05, 0.5)
        lr = []
        for i in range(len(jparams["fcLayers"])-1):
            lr_i =  trial.suggest_float("lr"+str(i), 1e-5, 0.1, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams['lr'].reverse()

        jparams["Optimizer"] = trial.suggest_categorical("Optimizer", ['SGD', 'Adam'])
        jparams["errorEstimate"] = trial.suggest_categorical("errorEstimate", ['one-sided', 'symmetric'])

        if jparams["Dropout"]:
            dropProb = []
            for i in range(len(jparams["fcLayers"]) - 1):
                drop_i = trial.suggest_float("drop" + str(i), 0.01, 1, log=True)
                # to verify whether we need to change the name of drop_i
                dropProb.append(drop_i)
            jparams["dropProb"] = dropProb.copy()
            jparams["dropProb"].reverse()

    elif jparams["action"] == 'supervised_ep':

        jparams["batchSize"] = trial.suggest_int("batchSize", 10, 256)
        jparams["eta"] = None
        jparams["gamma"] = None
        jparams["nudge_N"] = None
        jparams["beta"] = None
        lr = []
        for i in range(len(jparams["fcLayers"]) - 1):
            lr_i = trial.suggest_float("lr" + str(i), 1e-5, 0.1, log=True)
            # to verify whether we need to change the name of lr_i
            lr.append(lr_i)
        jparams["lr"] = lr.copy()
        jparams["lr"].reverse()

        jparams["Optimizer"] = trial.suggest_categorical("Optimizer", ['SGD', 'Adam'])
        jparams["errorEstimate"] = trial.suggest_categorical("errorEstimate", ['one-sided', 'symmetric'])
        jparams["lossFunction"] = trial.suggest_categorical("lossFunction", ['MSE', 'Cross-entropy'])

        if jparams["Dropout"]:
            dropProb = []
            for i in range(len(jparams["fcLayers"]) - 1):
                drop_i = trial.suggest_float("drop" + str(i), 0.01, 1, log=True)
                # to verify whether we need to change the name of drop_i
                dropProb.append(drop_i)
            jparams["dropProb"] = dropProb.copy()
            jparams["dropProb"].reverse()

    elif jparams["action"] == 'class_layer':
        jparams["batchSize"] = 128
        jparams["beta"] = 0.5
        jparams["gamma"] = 0.5
        jparams["errorEstimate"] = 'one-sided'
        jparams["lr"] = [0.01, 0.02]
        jparams["class_activation"] = trial.suggest_categorical("class_activation", ['softmax', 'sigmoid', 'hardsigm'])
        jparams["class_Optimizer"] = trial.suggest_categorical("class_Optimizer", ['Adam', 'SGD'])
        jparams["class_lr"] = trial.suggest_float("class_lr", 1e-4, 0.1, log=True)

    return jparams


def train_validation_test(jparams, net, trial, validation_loader, train_loader=None, class_loader=None, layer_loader=None, class_net=None):
    # train the model
    if jparams['action'] == 'supervised_ep':
        if train_loader is not None:
            print("Training the model with supervised ep")
        else:
            raise ValueError("training data is not given ")

        for epoch in tqdm(range(jparams['epochs'])):
            if jparams['lossFunction'] == 'MSE':
                train_error_epoch = train_supervised_ep(net, jparams, train_loader, epoch)
            elif jparams['lossFunction'] == 'Cross-entropy':
                train_error_epoch = train_supervised_crossEntropy(net, jparams, train_loader, epoch)

            validation_error_epoch = test_supervised_ep(net, validation_loader)

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
            Xth = train_unsupervised_ep(net, jparams, train_loader, epoch)
            # class process
            response, max0_indice = classify(net, jparams, class_loader)
            # test process
            error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, validation_loader, response)

            # Handle pruning based on the intermediate value.
            trial.report(error_av_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return error_av_epoch

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
        train_loader, validation_loader,  class_loader, layer_loader =\
            returnYinYang(jparams['batchSize'], batchSizeTest=jparams["test_batchSize"])
    elif jparams["dataset"] == 'mnist':
        train_loader,  validation_loader, class_loader, layer_loader =\
            returnMNIST(jparams["class_seed"], jparams["batchSize"], batchSizeTest=jparams["test_batchSize"])

    # create the model
    net = torch.jit.script(MlpEP(jparams))

    # load the trained unsupervised network when we train classification layer
    if jparams["action"] == 'class_layer':
        with open(r'C:\Users\CNRS-THALES\OneDrive\文档\Homeostasis_python\Eqprop-unsuperivsed-MLP\DATA-0\2023-03-02\S-6\model_entire.pt','rb') as f:
            loaded_net = torch.jit.load(f)
        net.W = loaded_net.W.copy()
        net.bias = loaded_net.bias.copy()
        net.eval()

        # create the new class_net
        class_net = Classlayer(jparams)

        final_err = train_validation_test(jparams, net, trial, validation_loader, layer_loader=layer_loader, class_net=class_net)

    elif jparams["action"] == 'unsupervised_ep':
        final_err = train_validation_test(jparams, net, trial, validation_loader, train_loader=train_loader, class_loader=class_loader)

    elif jparams["action"] == 'supervised_ep':
        final_err = train_validation_test(jparams, net, trial, validation_loader, train_loader=train_loader)

    del(jparams)

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
    study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db')
    study.optimize(lambda trial: objective(trial, pre_config), n_trials=100)

    trails = study.get_trials()
    # record trials
    df = study.trials_dataframe()
    df.to_csv(filePath)
    optuna.visualization.plot_param_importances(study)
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_slice(study)

    #np.savetxt(BASE_PATH + prefix + "test.csv", trails, delimiter=",", fmt='%s')
    #np.savetxt(BASE_PATH+"test.csv", trails, delimiter=",", fmt='%s', header=header)
    # save study and read the parameters in the study









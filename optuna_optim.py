import optuna
import torch
import numpy as np
import pandas
import torchvision
import argparse
from pathlib import Path
from Data import *
from Network_optuna import *
from Tools_optuna import *

parser = argparse.ArgumentParser(description='hyperparameter EP by optuna')
parser.add_argument(
    '--device',
    type=int,
    default=0,
    help='GPU name to use cuda')
parser.add_argument(
    '--structure',
    nargs='+',
    type=int,
    default=[784, 100],
    help='Test structure')
parser.add_argument(
    '--dataset',
    type=str,
    default='mnist',
    help='Dataset (default:YinYang, else:mnist)'
)
parser.add_argument(
    '--action',
    type=str,
    default='unsupervised_ep',
    help='Decide the learning method (default:supervised_ep, else:unsupervised_ep)'
)
parser.add_argument(
    '--epochs',
    type=int,
    default=40,
    metavar='N',
    help='number of epochs to train (default: 100)')
parser.add_argument(
    '--test_batchSize',
    type=int,
    default=128,
    help='Testing batch size (default=256)')
parser.add_argument(
    '--dt',
    type=float,
    default=0.2,
    help='time discretization (default: 0.2)')
parser.add_argument(
    '--exp_activation',
    type=str,
    default="hardsigm",
    help='activation function'
)
parser.add_argument(
    '--clamped',
    type=int,
    default=1,
    help='Clamped state of the network: crossed input are clamped to avoid divergence (default: True)')
parser.add_argument(
    '--n_class',
    type=int,
    default=10,
    help='the number of class (default = 10)'
)
parser.add_argument(
    '--Homeo_mode',
    type=str,
    default='batch',
    help='batch mode or SM mode'
)
parser.add_argument(
    '--exp_N',
    type=int,
    default=3,
    help='max N winner (default: 3)')
parser.add_argument(
    '--coeffDecay',
    type=float,
    default=1,
    help='the coefficient of learning rate Decay(default=1, other:0.5, 0.7)'
)
parser.add_argument(
    '--gammaDecay',
    type=float,
    default=1,
    help='the coefficient of Homeostasis Decay(default=1, other:0.5,0.6)'
)
parser.add_argument(
    '--epochDecay',
    type=float,
    default=100,
    help='the epoch to decay the learning rate (default=10, other:5, 15)'
)
parser.add_argument(
    '--weightNormalization',
    type=int,
    default=0,
    help='to decide whether to use the weight normalization'
)
parser.add_argument(
    '--Dropout',
    type=int,
    default=1,
    help='to decide whether to use the Dropout'
)
exp_args = parser.parse_args()

# define the activation function
if exp_args.exp_activation == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))

    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))

elif exp_args.exp_activation == 'hardsigm':
    def rho(x):
        return x.clamp(min=0).clamp(max = 1)

    def rhop(x):
        return (x >= 0) & (x <= 1)

elif exp_args.exp_activation == 'tanh':
    def rho(x):
        return torch.tanh(x)

    def rhop(x):
        return 1 - torch.tanh(x)**2


# define the dataset
def returnMNIST(class_seed, batchSize,batchSizeTest=256):
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

    # load the datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(validation_set, batch_size=batchSizeTest, shuffle=True)
    class_loader = torch.utils.data.DataLoader(class_set, batch_size=1000, shuffle=True)

    return train_loader, test_loader, class_loader


def returnYinYang(batchSize, batchSizeTest=128):
    print('We use the YinYang dataset')
    if exp_args.action == 'supervised_ep':
        train_set = YinYangDataset(size=5000, seed=42, target_transform=ReshapeTransformTarget(3))
    elif exp_args.action == 'unsupervised_ep':
        train_set = YinYangDataset(size=5000, seed=42)

    validation_set = YinYangDataset(size=1000, seed=41)  # used for the hyperparameter research
    class_set = YinYangDataset(size=1000, seed=41, sub_class=True)

    # test_set = YinYangDataset(size=1000, seed=40)
    # classTest_set = YinYangDataset(size=1000, seed=40, sub_class=True)

    # seperate the dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batchSize, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSizeTest, shuffle=False)
    # classTest_loader = torch.utils.data.DataLoader(classTest_set, batch_size=100, shuffle=False)

    class_loader = torch.utils.data.DataLoader(class_set, batch_size=100, shuffle=False)

    return train_loader, validation_loader, class_loader


def argsCreate(exp_args, batchSize, T, Kmax, beta, lr, Optimizer, errorEstimate, lossFunction, eta, gamma, nudge_N, dropProb):
    args = argparse.Namespace()
    args.device = exp_args.device
    args.dataset = exp_args.dataset
    args.action = exp_args.action
    args.epochs = exp_args.epochs
    args.batchSize = batchSize
    args.test_batchSize = exp_args.test_batchSize
    args.dt = exp_args.dt
    args.T = T
    args.Kmax = Kmax
    args.beta = beta
    args.clamped = 1
    args.convNet = 0
    args.convLayers = [1, 16, 5, 1, 1, 16, 32, 5, 1, 1]
    args.fcLayers = exp_args.structure.copy()
    args.lr = lr.copy()
    args.activation_function = exp_args.exp_activation
    args.Optimizer = Optimizer
    args.errorEstimate = errorEstimate
    args.lossFunction = lossFunction
    args.eta = eta
    args.gamma = gamma
    args.nudge_N = nudge_N
    args.n_class = exp_args.n_class

    args.coeffDecay = exp_args.coeffDecay
    args.gammaDecay = exp_args.gammaDecay
    args.epochDecay = exp_args.epochDecay
    args.weightNormalization = exp_args.weightNormalization
    args.Dropout = exp_args.Dropout
    args.dropProb = dropProb.copy()

    return args


def train_validation_test(args, net, trial, train_loader, validation_loader, class_loader):

    # train the model
    if exp_args.action == 'supervised_ep':
        print("Training the model with supervised ep")

        for epoch in tqdm(range(exp_args.epochs)):
            if args.lossFunction == 'MSE':
                train_error_epoch = train_supervised_ep(net, args, train_loader, epoch)
            elif args.lossFunction == 'Cross-entropy':
                train_error_epoch = train_supervised_crossEntropy(net, args, train_loader, epoch)

            validation_error_epoch = test_supervised_ep(net, args, validation_loader)

            # Handle pruning based on the intermediate value.
            trial.report(validation_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return validation_error_epoch

    elif args.action == 'unsupervised_ep':
        print("Training the model with unsupervised ep")

        for epoch in tqdm(range(args.epochs)):
            # train process
            Xth = train_unsupervised_ep(net, args, train_loader, epoch)
            # class process
            response, max0_indice = classify(net, args, class_loader)
            # test process
            error_av_epoch, error_max_epoch = test_unsupervised_ep(net, args, validation_loader, response)

            # Handle pruning based on the intermediate value.
            trial.report(error_av_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return error_av_epoch


def objective(trial, exp_args):

    # design the hyperparameters to be optimized
    if exp_args.Homeo_mode == 'SM' and exp_args.action == 'unsupervised_ep':
        batchSize = 1
    else:
        batchSize = trial.suggest_int("batchSize", 10, 256)
    #T = trial.suggest_categorical("T", [15, 20, 25])
    T = 40
    #Kmax = trial.suggest_categorical("Kmax", [5, 10])
    Kmax = 15
    beta = trial.suggest_float("beta", 0.05, 0.5)
    lr1 = trial.suggest_float("lr1", 1e-5, 0.1, log=True)

    if len(exp_args.structure) == 2:
        lr = [lr1]
    else:
        lr_coeff = trial.suggest_float("lr_coeff", 0.1, 5)
        lr = [lr_coeff*lr1, lr1]

    Optimizer = trial.suggest_categorical("Optimizer", ['SGD', 'Adam'])
    errorEstimate = trial.suggest_categorical("errorEstimate", ['one-sided', 'symmetric'])

    # Optimizer = 'Adam'
    # errorEstimate = 'symmetric'
    # lossFunction = 'Cross-entropy'
    if exp_args.action == 'supervised_ep':
        eta = 0.6
        gamma = 0.8
        nudge_N = 1
        lossFunction = trial.suggest_categorical("lossFunction", ['MSE', 'Cross-entropy'])
    else:
        gamma = trial.suggest_float("gamma", 0.001, 1, log=True)
        # TODO change the nudging number
        nudge_N = trial.suggest_int("nudge_N", 1, exp_args.exp_N)
        lossFunction = 'MSE'
        if exp_args.Homeo_mode == 'batch':
            eta = 0.6
        else:
            eta = trial.suggest_float("eta", 0.001, 1, log=True)
    if exp_args.dataset == 'mnist':
        class_seed = trial.suggest_int("class_seed", 1, 9)
    else:
        class_seed = 1

    if exp_args.dataset == 'YinYang' or exp_args.Dropout == 0:
        dropProb = [0, 0]
    else:
        drop1 = trial.suggest_float("drop1", 0.05, 0.4)
        drop2 = trial.suggest_float("drop2", 0.05, 0.5)
        dropProb = [drop1, drop2]

    # create the args for the training'
    args = argsCreate(exp_args, batchSize, T, Kmax, beta, lr, Optimizer, errorEstimate, lossFunction, eta, gamma, nudge_N, dropProb)
    args.fcLayers.reverse()  # we put in the other side, output first, input last
    args.lr.reverse()
    args.dropProb.reverse()

    # create the dataset
    if exp_args.dataset == 'YinYang':
        train_loader, validation_loader,  class_loader=\
            returnYinYang(batchSize, batchSizeTest=exp_args.test_batchSize)
    elif exp_args.dataset == 'mnist':
        train_loader,  validation_loader, class_loader=\
            returnMNIST(class_seed, batchSize, batchSizeTest=exp_args.test_batchSize)

    # create the model
    net = torch.jit.script(MlpEP(args))
    # training process
    final_err = train_validation_test(args, net, trial, train_loader, validation_loader, class_loader)

    return final_err


def saveHyperparameters(exp_args, BASE_PATH):
    '''
    Save all hyperparameters in the path provided
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    f = open(BASE_PATH + prefix + 'Hyperparameters.txt', 'w')
    f.write('Classic Equilibrium Propagation - Energy-based settings \n')
    f.write('   Parameters of the simulation \n ')
    f.write('\n')

    for key in exp_args.__dict__:
        f.write(key)
        f.write(': ')
        f.write(str(exp_args.__dict__[key]))
        f.write('\n')

    f.close()


def createHyperFile(parameters):

    Path('./HyperTest_optuna').mkdir(parents=True, exist_ok=True)
    path_hyper = Path(Path.cwd()/'HyperTest')

    filePathList = list(path_hyper.glob('*.csv'))

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
        if parameters['method'] == 'unsupervised_ep':
            columns_header = ['device', 'fcLayers', 'nudge_N', 'lr', 'alpha1', 'alpha2', 'alpha3', 'batch', 'T', 'Kmax', 'beta', 'gamma', 'eta', 'rho', 'final_epoch', 'final_Xth', 'one2one_av', 'one2one_max']
        elif parameters['method'] == 'supervised_ep':
            columns_header = ['device', 'fcLayers', 'nudge_N', 'lr', 'alpah1', 'alpha2', 'alpha3', 'batch', 'T', 'Kmax', 'beta', 'gamma', 'eta', 'rho',  'final_epoch', 'test_error']

        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(filePath)

    return dataframe, filePath, path_hyper


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
    # save hyperparameters
    saveHyperparameters(exp_args, BASE_PATH)
    # create the filepath for saving the optuna trails
    filePath = BASE_PATH + prefix + "test.csv"
    study_name = str(time.asctime())
    study = optuna.create_study(study_name=study_name, storage='sqlite:///example.db')
    study.optimize(lambda trial: objective(trial, exp_args), n_trials=100)
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









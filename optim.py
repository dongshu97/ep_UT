import nevergrad as ng
import numpy as np
import torch
import torchvision
import argparse
import pathlib
from pathlib import Path
import pandas as pd
from Data import *
from Network_para import *
from Tools_para import *

parser = argparse.ArgumentParser(description='hyperparameter EP')
parser.add_argument(
    '--structure',
    nargs='+',
    type=int,
    default=[784, 1024],
    help='Test structure')
parser.add_argument(
    '--Homeo_mode',
    type=str,
    default='SM',
    help='batch mode or SM mode'
)
parser.add_argument(
    '--exp_N',
    type=int,
    default=6,
    help='N winner (default: 1)')
parser.add_argument(
    '--exp_activation',
    type=str,
    default='hardsigm',
    help='N winner (default: 1)')
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


def returnMNIST(batchSize,  conv, batchSizeTest=256):

    # define the optimization dataset
    print('We use the MNIST Dataset')
    # Define the Transform
    # !! Attention it depends on whether use the convolutional layers

    if conv == 0:
        transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]
    else:
        transforms = [torchvision.transforms.ToTensor()]

    # Down load the MNIST dataset
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                           transform=torchvision.transforms.Compose(transforms),
                                           target_transform=ReshapeTransformTarget(10))

    rest_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    validation_seed = 1

    validation_set = ValidationDataset(root='./MNIST_validate_seed', rest_set=rest_set, seed=validation_seed,
                             transform=torchvision.transforms.Compose(transforms))

    classValidation_set = ClassDataset(root='./MNIST_classValidate', test_set=validation_set, seed=validation_seed,
                             transform=torchvision.transforms.Compose(transforms))

    test_set = HypertestDataset(root='./MNIST_validate_seed', rest_set=rest_set, seed=validation_seed,
                             transform=torchvision.transforms.Compose(transforms))

    classTest_set = ClassDataset(root='./MNIST_classTest', test_set=test_set, seed=validation_seed,
                                       transform=torchvision.transforms.Compose(transforms))

    # load the datasets
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batchSize, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=batchSizeTest, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batchSizeTest, shuffle=True)
    classValidation_loader = torch.utils.data.DataLoader(classValidation_set, batch_size=300, shuffle=True)
    classTest_loader = torch.utils.data.DataLoader(classTest_set, batch_size=700, shuffle=True)

    return train_loader, validation_loader, test_loader, classValidation_loader, classTest_loader


def argsCreate(device, epoch, batchSize, T, Kmax, beta, lr, eta, gamma, nudge_N, convNet, fcLayers, rho, dt, method='bp_Xth'):
    args = argparse.Namespace()
    args.device = device
    args.dataset = "mnist"
    args.action = method
    args.epochs = epoch
    args.batchSize = batchSize
    args.test_batchSize = 256
    args.dt = dt
    args.T = T
    args.Kmax = Kmax
    args.beta = beta
    args.clamped = 1
    args.convNet = convNet
    args.convLayers = [1, 16, 5, 1, 1, 16, 32, 5, 1, 1]
    args.fcLayers = fcLayers.copy()
    args.lr = lr.copy()
    args.activation_function = rho
    args.eta = eta
    args.gamma = gamma
    args.nudge_N = nudge_N
    args.n_class = 10

    return args


def training(device:int, lr:float, alpha1:float, alpha2:float, alpha3:float, batchSize:int, T:int, Kmax:int, beta:float, gamma:float, eta:float, epoch:int, method:str, convNet:int, fcLayers:list, rho:str, nudge_N:int):

    # We load the dataset
    train_loader, validation_loader, test_loader, \
    classValidation_loader, classTest_loader = returnMNIST(batchSize, conv=convNet)

    # define lr list depending on different network structure
    if method == 'unsupervised_conv_ep' or method == 'supervised_conv_ep':
        lr_list = [lr*alpha1, lr*alpha2, lr*alpha3, lr]

    elif method == 'supervised_ep' or method == 'unsupervised_ep':
        if len(fcLayers) == 2:
            lr_list = [lr]
        elif len(fcLayers) == 3:
            lr_list = [lr*alpha1, lr]

    args = argsCreate(device=device, epoch=epoch, batchSize=batchSize, T=T, Kmax=Kmax,
                      beta=beta, lr=lr_list,
                      eta=eta, gamma=gamma, nudge_N=nudge_N, convNet=convNet,
                      fcLayers=fcLayers, rho=rho, dt=0.2, method=method)

    args.fcLayers.reverse()  # we put in the other side, output first, input last
    args.lr.reverse()
    # TODO if we add the rhoLayers, we have to reverse the rhoLayers here too

    # we train 3 times for each parameters

    # TODO add the moyenne time in the hyper-parameters
    moyenne_times = 1
    moyenne_av = 0
    moyenne_max = 0
    moyenne_epoch = 0
    moyenne_Xth = 0
    moyenne_supervised = 0

    for m in range(moyenne_times):
        # we create the network structure
        if args.convNet:
            net = ConvEP(args)
        else:
            net = MlpEP(args)

        if method == 'unsupervised_ep':

            validation_error_av = []
            validation_error_max = []

            Xth = torch.zeros(args.fcLayers[0], device=net.device)

            EA_av, EA_max = -1, -1

            clock = 0

            for epoch in range(args.epochs):

                Xth = train_unsupervised_ep(net, args, train_loader, epoch, Xth)

                # classifying process
                response,  max0_indice = classify(net, args, classValidation_loader)

                # testing process
                error_av_epoch, error_max_epoch = test_unsupervised_ep(net, args, validation_loader, response)

                validation_error_av.append(error_av_epoch.item())
                validation_error_max.append(error_max_epoch.item())

                one2one_av_min = min(validation_error_av)
                one2one_max_min = min(validation_error_max)

                if epoch > 5:
                    EA_av_new = np.array(validation_error_av[-5:]).mean()
                    EA_max_new = np.array(validation_error_max[-5:]).mean()

                    if EA_av > 0 and EA_max > 0:
                        if EA_av_new >= EA_av and EA_max_new >= EA_max:
                            clock += 1

                    EA_av = EA_av_new
                    EA_max = EA_max_new

                if clock > 2:
                    break

                if error_av_epoch > one2one_av_min and error_max_epoch > one2one_max_min:
                #if error_max_epoch > one2one_max_min * 1.03:
                    break

            response, max0_indice = classify(net, args, classTest_loader)
            test_error_av, test_error_max = test_unsupervised_ep(net, args, test_loader, response=response)
            Xth_norm = torch.norm(Xth).item()
            print('The training finish at Epoch:', epoch)
            moyenne_av += float(test_error_av)
            moyenne_max += float(test_error_max)
            moyenne_Xth += float(Xth_norm)
            moyenne_epoch += epoch

        elif method == 'supervised_ep':

            validation_train_error = []
            validation_test_error = []

            EA_test, EA_train = -1, -1
            clock = 0

            for epoch in range(args.epochs):

                train_error_epoch = train_supervised_ep(net, args, train_loader, epoch)

                # testing process
                test_error_epoch = test_supervised_ep(net, args, validation_loader)

                validation_train_error.append(train_error_epoch.item())
                validation_test_error.append(test_error_epoch.item())

                validation_train_min = min(validation_train_error)
                validation_test_min = min(validation_test_error)

                if epoch > 5:
                    EA_train_new = np.array(validation_train_error[-5:]).mean()
                    EA_test_new = np.array(validation_test_error[-5:]).mean()

                    if EA_test > 0 and EA_train > 0:
                        if EA_train_new >= EA_train and EA_test_new >= EA_test:
                            clock += 1

                    EA_train = EA_train_new
                    EA_test = EA_test_new

                if clock > 2:
                    break

                if train_error_epoch > validation_train_min * 1.05 and test_error_epoch > validation_test_min * 1.05:
                    # if error_max_epoch > one2one_max_min * 1.03:
                    break

            test_error_final = test_supervised_ep(net, args, test_loader)

            #test_error_av, test_error_max = test_Xth(net, args, test_loader, response=response)

            print('The training finish at Epoch:', epoch)

            moyenne_supervised += float(test_error_final)
            moyenne_epoch += epoch

    if method == 'unsupervised_ep':
        moyenne_av = moyenne_av /moyenne_times
        moyenne_max = moyenne_max / moyenne_times
        moyenne_Xth = moyenne_Xth / moyenne_times
        moyenne_epoch = moyenne_epoch / moyenne_times

        return moyenne_av, moyenne_max, moyenne_epoch, moyenne_Xth

    elif method == 'supervised_ep':
        moyenne_supervised = moyenne_supervised / moyenne_times
        moyenne_epoch = moyenne_epoch / moyenne_times

        return moyenne_supervised, moyenne_epoch


def createHyperFile(parameters):

    Path('./HyperTest').mkdir(parents=True, exist_ok=True)
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


def updataDataframe(filePath, dataframe, parameters, epoch, Xth=None, one2one_av=None, one2one_max=None, test_error=None):

    if parameters['method'] == 'unsupervised_ep':
        data = [parameters['device'], parameters['fcLayers'], parameters['nudge_N'], parameters['lr'], parameters['alpha1'], parameters['alpha2'], parameters['alpha3'], parameters['batchSize'], parameters['T'], parameters['Kmax'], parameters['beta'], parameters['gamma'], parameters['eta'], parameters['rho'], epoch, Xth, one2one_av, one2one_max]
    elif parameters['method'] == 'supervised_ep':
        data = [parameters['device'], parameters['fcLayers'], parameters['nudge_N'], parameters['lr'], parameters['alpha1'], parameters['alpha2'], parameters['alpha3'], parameters['batchSize'], parameters['T'], parameters['Kmax'], parameters['beta'], parameters['gamma'], parameters['eta'], parameters['rho'], epoch, test_error]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(filePath)
    except PermissionError:
        input("Close the result.csv and press any key.")

    return dataframe


if __name__ == '__main__':

    if exp_args.Homeo_mode == 'batch' and len(exp_args.structure) == 3:
        # TODO to be changed
        parametrization = ng.p.Instrumentation(
            device=0,
            lr=ng.p.Log(lower=0.001, upper=0.01),
            #lr=ng.p.Choice([0.009, 0.0095, 0.01, 0.012, 0.013, 0.014, 0.015, 0.016, 0.018, 0.02]),
            #alpha1=ng.p.Log(lower=0.5, upper=3),
            alpha1=ng.p.Choice([0.5, 1, 1.5, 2, 2.25, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 3.8, 4, 4.15, 4.3, 4.5, 4.8, 5, 5.5]),
            #alpha2=ng.p.Log(lower=0.1, upper=10),
            alpha2=1,
            #alpha3=ng.p.Scalar(lower=0.0, upper=1.0),
            alpha3=1,
            batchSize=ng.p.Scalar(lower=40, upper=200).set_integer_casting(),
            T=60,
            Kmax=20,
            beta=ng.p.Choice([0.12, 0.15, 0.18, 0.2, 0.22, 0.25, 0.3, 0.35, 0.38, 0.4, 0.45, 0.48, 0.5]),
            #beta=ng.p.Choice([0.3, 0.35, 0.38, 0.4, 0.42, 0.45, 0.5]),
            gamma=ng.p.Log(lower=0.1, upper=1.0),
            #gamma=ng.p.Choice([0.6, 0.8, 0.9]),
            eta=0.6,
            epoch=200,
            method='unsupervised_ep',
            convNet=0,
            fcLayers=exp_args.structure,
            rho='relu',
            #rho=ng.p.Choice(['hardsigm', 'tanh']),
            #nudge_N=exp_args.exp_N
            nudge_N=ng.p.Choice([3, 4, 5])
            # coeff_Decay=ng.p.Choice([0.5, 0.6, 0.7, 0.8, 0.9]),
            # epoch_Decay=ng.p.Choice([50, 60, 70, 80, 100, 150])
        )
    elif exp_args.Homeo_mode == 'batch' and len(exp_args.structure) == 2:
        parametrization = ng.p.Instrumentation(
            device=0,
            lr=ng.p.Log(lower=0.001, upper=0.1),
            alpha1=1,
            # alpha2=ng.p.Log(lower=0.1, upper=10),
            alpha2=1,
            # alpha3=ng.p.Scalar(lower=0.0, upper=1.0),
            alpha3=1,
            batchSize=ng.p.Scalar(lower=128, upper=256).set_integer_casting(),
            T=40,
            Kmax=15,
            beta=ng.p.Choice([0.05, 0.1, 0.15, 0.2, 0.3, 0.5]),
            gamma=ng.p.Log(lower=0.05, upper=1.0),
            eta=0.6,
            epoch=50,
            method='unsupervised_ep',
            convNet=0,
            fcLayers=exp_args.structure,
            rho='hardsigm',
            # rho=ng.p.Choice(['hardsigm', 'tanh']),
            nudge_N=exp_args.exp_N
        )
    elif exp_args.Homeo_mode != 'batch' and len(exp_args.structure) == 3:
        parametrization = ng.p.Instrumentation(
            device=-1,
            lr=ng.p.Log(lower=0.00001, upper=0.001),
            alpha1=ng.p.Choice([0.5, 0.6, 0.8, 1, 1.2, 1.5, 2, 2.25, 2.4, 2.5, 2.6, 2.8, 3, 3.25, 3.5, 4]),
            # alpha2=ng.p.Log(lower=0.1, upper=10),
            alpha2=1,
            # alpha3=ng.p.Scalar(lower=0.0, upper=1.0),
            alpha3=1,
            batchSize=1,
            T=60,
            Kmax=20,
            beta=ng.p.Choice([0.01, 0.03, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]),
            gamma=ng.p.Log(lower=0.05, upper=1.0),
            eta=ng.p.Log(lower=0.005, upper=1.0),
            epoch=20,
            method='unsupervised_ep',
            convNet=0,
            fcLayers=exp_args.structure,
            rho='hardsigm',
            # rho=ng.p.Choice(['hardsigm', 'tanh']),
            nudge_N=ng.p.Choice([1])
        )
    elif exp_args.Homeo_mode != 'batch' and len(exp_args.structure) == 2:
        parametrization = ng.p.Instrumentation(
            device=-1,
            lr=ng.p.Log(lower=0.00001, upper=0.001),
            alpha1=1,
            # alpha2=ng.p.Log(lower=0.1, upper=10),
            alpha2=1,
            # alpha3=ng.p.Scalar(lower=0.0, upper=1.0),
            alpha3=1,
            batchSize=1,
            T=40,
            Kmax=10,
            beta=ng.p.Choice([0.05, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.5]),
            gamma=ng.p.Log(lower=0.05, upper=1.0),
            eta=ng.p.Log(lower=0.005, upper=1.0),
            epoch=10,
            method='unsupervised_ep',
            convNet=0,
            fcLayers=exp_args.structure,
            rho='hardsigm',
            # rho=ng.p.Choice(['hardsigm', 'tanh']),
            nudge_N=ng.p.Choice([5, 6])
        )

    #optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=50, num_workers=2)
    #TODO change the budget back to 1
    optimizer = ng.optimizers.PortfolioDiscreteOnePlusOne(parametrization=parametrization, budget=50, num_workers=2)
    #optimizer = ng.optimizers.RandomSearch(parametrization=parametrization, budget=50, num_workers=2)
    # optim.suggest(lr=0.0005, batch_size=40, gamma=0.1)

    #average_N = 1

    x = optimizer.ask()

    dataframe, filePath, path_hyper = createHyperFile(x.kwargs)

    method = x.kwargs['method']

    optimizedFile = 'initial' + Path(filePath).stem + '.txt'

    with open(path_hyper / optimizedFile, 'w') as f:
        for key, value in x.kwargs.items():
            f.write('%s:%s\n' % (key, value))
        f.write('%s:%s\n' % (key, value))

    if method == 'unsupervised_ep':

        for _ in range(optimizer.budget):
            x1 = optimizer.ask()
            x2 = optimizer.ask()

        #print('kwargs of x1 is', x1.kwargs)
        #print('kwargs of x2 is', x2.kwargs)
        # av_avError, av_maxError, av_epoch = 0
        # for av in range(average_N):
        #     one2one_av, one2one_max, epoch = training(*x.args, **x.kwargs)
        #     print('one2one_av is', one2one_av, 'one2one_max is', one2one_max)
        #     av_avError += one2one_av
        #     av_maxError += one2one_max
        #     av_epoch += epoch
        # av_avError = av_avError/average_N
        # av_maxError = av_maxError/average_N
        # av_epoch = av_epoch/average_N
            one2one_av1, one2one_max1, epoch1, Xth1 = training(*x1.args, **x1.kwargs)
            one2one_av2, one2one_max2, epoch2, Xth2 = training(*x2.args, **x2.kwargs)
            #print('one2one_av1 is', one2one_av1, 'one2one_max1 is', one2one_max1)
            #print('one2one_av2 is', one2one_av2, 'one2one_max2 is', one2one_max2)
            #optimizer.tell(x1, one2one_max1)
            #optimizer.tell(x2, one2one_max2)
            optimizer.tell(x1, one2one_av1)
            optimizer.tell(x2, one2one_av2)

            dataframe = updataDataframe(filePath, dataframe, x1.kwargs, epoch1, Xth=Xth1, one2one_av=one2one_av1, one2one_max=one2one_max1)
            dataframe = updataDataframe(filePath, dataframe, x2.kwargs, epoch2, Xth=Xth2, one2one_av=one2one_av2, one2one_max=one2one_max2)

    elif method == 'supervised_ep':

        for _ in range(optimizer.budget):
            x1 = optimizer.ask()
            x2 = optimizer.ask()
            test_error1, epoch1 = training(*x1.args, **x1.kwargs)
            test_error2, epoch2 = training(*x2.args, **x2.kwargs)
            dataframe = updataDataframe(filePath, dataframe, x1.kwargs, epoch1, test_error=test_error1)
            dataframe = updataDataframe(filePath, dataframe, x2.kwargs, epoch2, test_error=test_error2)

    recommendation = optimizer.recommend()
    print('The recommendation kwargs is', recommendation.kwargs)

    optimizedFile = 'optimized'+Path(filePath).stem + '.txt'
    with open(path_hyper/optimizedFile, 'w') as f:
        for key, value in recommendation.kwargs.items():
            f.write('%s:%s\n' % (key, value))
        f.write('%s:%s\n' % (key, value))
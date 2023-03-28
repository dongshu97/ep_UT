#Main for the simulation
import os
import argparse  # this can be removed
import json
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torchvision
import torch.optim as optim
import pickle
import datetime
import numpy as np
import platform
import pathlib
import time
from tqdm import tqdm

from Data import *
from Tools import *
from Network import *
from plotFunction import*
from visu import *

# TODO use argsparser to give the file path of json file
parser = argparse.ArgumentParser(description='Path of json file')
parser.add_argument(
    '--json_path',
    type=str,
    default=r'.',
    help='path of json configuration'
)
args = parser.parse_args()
# parser = argparse.ArgumentParser(description='usupervised EP')
# parser.add_argument(
#     '--device',
#     type=int,
#     default=0,
#     help='GPU name to use cuda')
# parser.add_argument(
#     '--dataset',
#     type=str,
#     default="mnist",
#     help='dataset to be used to train the network : (default = mnist)')
# # TODO
# # parser.add_argument(
# #     '--training-method',
# #     type=str,
# #     default='ep',
# #     help='traning method (default:ep)'
# # )
# parser.add_argument(
#     '--action',
#     type=str,
#     default="unsupervised_ep",
#     help='train or test: (default = unsupervised_ep, other: supervised_ep, test, visu')
# parser.add_argument(
#     '--epochs',
#     type=int,
#     default=40,
#     metavar='N',
#     help='number of epochs to train (default: 100)')
# parser.add_argument(
#     '--batchSize',
#     type=int,
#     default=118,
#     help='Batch size (default=128)')
# parser.add_argument(
#     '--test_batchSize',
#     type=int,
#     default=512,
#     help='Testing batch size (default=256)')
# parser.add_argument(
#     '--dt',
#     type=float,
#     default=0.2,
#     help='time discretization (default: 0.2)')
# parser.add_argument(
#     '--T',
#     type=int,
#     default=60,
#     help='number of time steps in the free phase (default: 40) - Let the system relax with oscillators dynamics')
# parser.add_argument(
#     '--Kmax',
#     type=int,
#     default=20,
#     help='number of time steps in the backward pass (default: 50)')
# parser.add_argument(
#     '--beta',
#     type=float,
#     default=0.5,
#     metavar='BETA',
#     help='nudging parameter (default: 0.5)')
# parser.add_argument(
#     '--clamped',
#     type=int,
#     default=1,
#     help='Clamped state of the network: crossed input are clamped to avoid divergence (default: True)')
# parser.add_argument(
#     '--convNet',
#     type=int,
#     default=0,
#     help='whether use the convolutional layers'
# )
# # parser.add_argument(
# #     '--convLayers',
# #     nargs='+',
# #     type=int,
# #     default=[1, 16, 5, 1, 1, 16, 32, 5, 1, 1],
# #     help='The parameters of convNet, each conv layer has 5 parameter: in_channels, out_channels, K/F, S, P')
# # TODO rewrite the argparse for the CNN
# # we suppose the stride=1
# parser.add_argument(
#     '--C_list',
#     nargs='+',
#     type=int,
#     default=[1, 32, 64],
#     help='channel list (default: [])')
# parser.add_argument(
#     '--padding',
#     type=int,
#     default=0,
#     help='padding or not (default: 0, else:1)')
# parser.add_argument(
#     '--convF',
#     type=int,
#     default=5,
#     metavar='F',
#     help='convolution filter size (default: 5)')
# parser.add_argument(
#     '--Fpool',
#     type=int,
#     default=2,
#     metavar='Fp',
#     help='pooling filter size (default: 2)')
# parser.add_argument(
#     '--fcLayers',
#     nargs='+',
#     type=int,
#     default=[784, 1024, 512],
#     help='The parameters of convNet, each conv layer has 5 parameter: in_channels, out_channels, K/F, S, P')
# parser.add_argument(
#     '--lr',
#     nargs='+',
#     type=float,
#     default=[0.0107, 0.00257],
#     help='learning rates')
# parser.add_argument(
#     '--activation_function',
#     type=str,
#     default='hardsigm',
#     help='activation function')
# parser.add_argument(
#     '--Optimizer',
#     type=str,
#     default='SGD',
#     help='the optimizer to be used (default=SGD, else:Adam)'
# )
# parser.add_argument(
#     '--errorEstimate',
#     type=str,
#     default='symmetric',
#     help='Two different way to estimate the loss (default=one-sided, else:symmetric)'
# )
# parser.add_argument(
#     '--lossFunction',
#     type=str,
#     default='MSE',
#     help='Define the type of loss function (default=MSE, else:Cross-entropy)'
# )
# parser.add_argument(
#     '--eta',
#     type=float,
#     default=0.2,
#     help='the coefficient for regulating the homeostasis effect (default: 0.1)'
# )
# parser.add_argument(
#     '--gamma',
#     type=float,
#     default=0.668,
#     help='the coefficient for regulating the homeostasis effect (default: 0.2)'
# )
# parser.add_argument(
#     '--nudge_N',
#     type=int,
#     default=5,
#     help='the number of winners to be nudged (default: 1)'
# )
# parser.add_argument(
#     '--n_class',
#     type=int,
#     default=10,
#     help='the number of class (default = 10)'
# )
# parser.add_argument(
#     '--class_activation',
#     type=str,
#     default='softmax',
#     help='activation function of the added classification layer')
# parser.add_argument(
#     '--class_lr',
#     type=float,
#     default=0.045,
#     help='the learning rate of the classification layer'
# )
# parser.add_argument(
#     '--class_epoch',
#     type=int,
#     default=30,
#     help='the epoch for training the classification layer'
# )
# parser.add_argument(
#     '--class_Optimizer',
#     type=str,
#     default='Adam',
#     help='the Optimizer used for the classification layer'
# )
# parser.add_argument(
#     '--coeffDecay',
#     type=float,
#     default=1,
#     help='the coefficient of learning rate Decay(default=1, other:0.5, 0.7)'
# )
# parser.add_argument(
#     '--gammaDecay',
#     type=float,
#     default=1,
#     help='the coefficient of Homeostasis Decay(default=1, other:0.5,0.6)'
# )
# parser.add_argument(
#     '--epochDecay',
#     type=float,
#     default=1,
#     help='the epoch to decay the learning rate (default=10, other:5, 15)'
# )
# parser.add_argument(
#     '--weightNormalization',
#     type=int,
#     default=0,
#     help='to decide whether to use the weight normalization'
# )
# parser.add_argument(
#     '--randomHidden',
#     type=int,
#     default=0,
#     help='update only the weights of output layers'
# )
# parser.add_argument(
#     '--Dropout',
#     type=int,
#     default=0,
#     help='to decide whether to use the Dropout'
# )
# parser.add_argument(
#     '--dropProb',
#     nargs='+',
#     type=float,
#     default=[0, 0],
#     help='to decide the probability of dropout'
# )
# parser.add_argument(
#     '--torchSeed',
#     type=int,
#     default=0,
#     help='to generate the reproductive result'
# )
# parser.add_argument(
#     '--analysis_preTrain',
#     type=str,
#     default=0,
#     help='whether to load the trained model'
# )
# parser.add_argument(
#     '--imWeights',
#     type=int,
#     default=0,
#     help='whether we imshow the weights of synapses'
# )
# parser.add_argument(
#     '--maximum_activation',
#     type=int,
#     default=0,
#     help='draw the maximum activation input for each neuron'
# )
# parser.add_argument(
#     '--imShape',
#     nargs='+',
#     type=int,
#     default=[28, 28, 32, 32],
#     help='decide the size for each imshow of weights'
# )
# parser.add_argument(
#     '--display',
#     nargs='+',
#     type=int,
#     default=[10, 10, 10, 10],
#     help='decide the number of neurons whose weights are presented'
# )
# # input the args
# args = parser.parse_args()

# use json to load the configuration parameters
if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'

with open(args.json_path + prefix + 'config.json') as f:
  jparams = json.load(f)

#print(jparams)

# define the torchSeed
if jparams['torchSeed']:
    torch.manual_seed(jparams['torchSeed'])

# define the two batch sizes
batch_size = jparams['batchSize']
batch_size_test = jparams['test_batchSize']

# if args.dataset == 'digits':
#
#     print('We use the DIGITS Dataset')
#     from sklearn.datasets import load_digits
#     from sklearn.model_selection import train_test_split
#
#     digits = load_digits()
#
#     # TODO make the class_seed of digits dataset
#     x_total, x_class, y_total, y_class = train_test_split(digits.data, digits.target, test_size=0.1, random_state=0,
#                                                           shuffle=True)
#     x_train, x_test, y_train, y_test = train_test_split(x_total, y_total, test_size=0.15, random_state=0, shuffle=True)
#
#     x_class, x_train, x_test = x_class/16, x_train/16, x_test/16  # 0 to 1
#
#     class_data = DigitsDataset(x_class, labels=y_class, target_transforms=ReshapeTransformTarget(10))
#     train_data = DigitsDataset(x_train, labels=y_train, target_transforms=ReshapeTransformTarget(10))
#     test_data = DigitsDataset(x_test, labels=y_test, target_transforms=ReshapeTransformTarget(10))
#
#     len_class = len(x_class[:])
#
#     # dataloaders
#     class_loader = torch.utils.data.DataLoader(class_data, batch_size=batch_size, shuffle=True)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_test, shuffle=False)


if jparams['dataset'] == 'mnist':
    print('We use the MNIST Dataset')
    # Define the Transform
    if jparams['convNet']:
        transforms = [torchvision.transforms.ToTensor()]
    else:
        transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

    # Download the MNIST dataset
    if jparams['action'] == 'unsupervised_ep' or jparams['action'] == 'unsupervised_conv_ep' or jparams['action'] == 'test':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms))
        # # we create the dataset mixed with unlabeled and labeled data
        # if args.unlabeledPercent != 0 and args.unlabeledPercent != 1:
        #     train_set = UnlabelDataset(train_set, './MNIST_alterLearning', args.unlabeledPercent, Seed=0)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)
    else:
    #elif jparams['action'] == 'supervised_ep' or jparams['action'] == 'train_conv_ep':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms),
                                               target_transform=ReshapeTransformTarget(10))
        # if args.unlabeledPercent != 0 and args.unlabeledPercent != 1:
        #     train_set = subMNISTDataset(root='./MNIST_alterLearning', train_set=train_set,
        #                                 unlabeledPercent=args.unlabeledPercent,
        #                                 Seed=0, transform=torchvision.transforms.Compose(transforms),
        #                                 target_transform=ReshapeTransformTarget(10))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=jparams['test_batchSize'], shuffle=True)

    # define the class dataset
    seed = 34  # seed number should between 0 to 42

    # TODO this part can use the same method as data-split of semi-supervised learning
    x = train_set.data
    y = train_set.targets

    label_percentage = jparams['label_percentage']
    if jparams['label_percentage']==1:
        class_set = train_set
        layer_set = train_set
    else:
        class_set = splitClass(x, y, label_percentage, seed=seed, transform=torchvision.transforms.Compose(transforms))
    #
    # class_set = ClassDataset(root='./MNIST_class_seed', test_set=test_set, seed=seed,
    #                          transform=torchvision.transforms.Compose(transforms))

        layer_set = splitClass(x, y, label_percentage, seed=seed, transform=torchvision.transforms.Compose(transforms),
                                 target_transform=ReshapeTransformTarget(10))

    # layer_set = ClassDataset(root='./MNIST_class_seed', test_set=test_set, seed=seed,
    #                          transform=torchvision.transforms.Compose(transforms),
    #                              target_transform=ReshapeTransformTarget(10))

    if jparams['device'] >= 0:
        class_loader = torch.utils.data.DataLoader(class_set, batch_size=1200, shuffle=True)
    else:
        class_loader = torch.utils.data.DataLoader(class_set, batch_size=jparams['test_batchSize'], shuffle=True)

    layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=1200, shuffle=True)

elif jparams['dataset'] == 'YinYang':
    print('We use the YinYang dataset')
    ''' This dataset is not considered to apply with a ConvNet'''
    if jparams['convNet']:
        raise ValueError("YinYang dataset is not designed for ConvNet, the value should not be " "but got {}".format(jparams['convNet']))
    else:
        if jparams['action'] == 'supervised_ep':
            train_set = YinYangDataset(size=5000, seed=42, target_transform=ReshapeTransformTarget(3))
        elif jparams['action'] == 'unsupervised_ep' or jparams['action'] == 'test' or jparams['action'] == 'visu':
            train_set = YinYangDataset(size=5000, seed=42)

        #validation_set = YinYangDataset(size=1000, seed=41)  # used for the hyperparameter research

        test_set = YinYangDataset(size=1000, seed=40)

        class_set = YinYangDataset(size=1000, seed=42, sub_class=True)

        layer_set = YinYangDataset(size=1000, seed=42, target_transform=ReshapeTransformTarget(3), sub_class=True)

        # seperate the dataset
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=jparams['batchSize'], shuffle=True)
        #validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batchSize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=jparams['test_batchSize'], shuffle=False)
        class_loader = torch.utils.data.DataLoader(class_set, batch_size=100, shuffle=True)
        layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=100, shuffle=True)


#  TODO finish the CIFAR 10 dataset
elif jparams['dataset'] == 'CIFAR10':
    print('We use the CIFAR10 Dataset')
    # TODO to complete the CIFAR10


# define the activation function
if jparams['activation_function'] == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))

    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))

elif jparams['activation_function'] == 'hardsigm':
    def rho(x):
        return x.clamp(min=0).clamp(max=1)

    def rhop(x):
        return (x >= 0) & (x <= 1)

elif jparams['activation_function'] == 'tanh':
    def rho(x):
        return torch.tanh(x)

    def rhop(x):
        return 1 - torch.tanh(x)**2

elif jparams['activation_function'] == 'relu':
    def rho(x):
        return x.clamp(min=0)

    def rhop(x):
        return (x>=0)

if __name__ == '__main__':

    jparams['fcLayers'].reverse()  # we put in the other side, output first, input last
    jparams['C_list'].reverse()  # we reverse also the list of channels
    jparams['lr'].reverse()
    jparams['display'].reverse()
    jparams['imShape'].reverse()
    jparams['dropProb'].reverse()
    jparams['pruneAmount'].reverse()

    BASE_PATH, name = createPath()

    # we create the network
    if jparams['convNet']:
        net = ConvEP(jparams)
    else:
        net = torch.jit.script(MlpEP(jparams))

    # we load the pre-trained network
    if jparams['analysis_preTrain']:
        with open(r'C:/model_entire.pt', 'rb') as f:
            net = torch.jit.load(f)
        net.eval()

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # save hyper-parameters as json file
    with open(BASE_PATH + prefix + "config.json", "w") as outfile:
        json.dump(jparams, outfile)

    # TODO remove the saveHyperparameters(args, BASE_PATH)

    if jparams['action'] == 'test':
        print("Testing the model")

        # saveHyperparameters(args, BASE_PATH)

        DATAFRAME = initDataframe(BASE_PATH, method='supervised')
        print(DATAFRAME)

        data, target = next(iter(train_loader))

        s = net.initState(data)

        # TODO rewrite the registation for dy
        if net.cuda:
            s = [item.to(net.device) for item in s]
            target = target.to(net.device)

        s, y, h = net.forward(s, tracking=True)

        seq = s.copy()

        #unsupervised_target, maxindex = net.unsupervised_target(s[0].detach(), 1)

        #s, y1, h1 = net.forward(s, target=unsupervised_target, beta=0.5, tracking=True)

        s, y1, h1 = net.forward(s, target=target, beta=0.5, tracking=True)

        gradW, gradBias = net.computeGradientsEP(s, seq)
        print(gradW[0])

        # update and track the weights of the network
        net.updateWeight(s, seq, jparams['beta'])

        fig = plt.figure()
        for k in range(len(y)):
            plt.plot(y[k]+y1[k], label=f'Output{k}')
            # TODO to be changed back
            #plt.plot(h[k]+h1[k], '--', label='hidden layer')
        #plt.legend(loc='upper left', ncol=1, fontsize=6, frameon=False)
        plt.xlabel('Time steps')
        #plt.yticks(np.arange(0, 1, step=0.2))
        plt.ylabel('Different neuron values')
        plt.axvline(0, ymin=0, ymax=0.8, ls='--', linewidth=1, color='b')
        plt.axvline(jparams['T']-1, ymin=0, ymax=0.8, ls='--', linewidth=1, color='b')
        plt.axvline(jparams['T'] + jparams['Kmax']-1, ymin=0, ymax=0.8, ls='--', linewidth=1, color='b')
        plt.title('Dynamics of Equilibrium Propagation')
        fig.savefig(BASE_PATH + prefix + 'dymanics.png', format='png', dpi=300)
        plt.show()

    elif jparams['action'] == 'supervised_ep':
        print("Training the model with supervised ep")

        #saveHyperparameters(args, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, method='supervised')

        # save the initial network
        torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict_0.pt')

        train_error_list = []
        test_error_list = []

        for epoch in tqdm(range(jparams['epochs'])):
            if jparams['lossFunction'] == 'MSE':
                train_error_epoch = train_supervised_ep(net, jparams, train_loader, epoch)
            elif jparams['lossFunction'] == 'Cross-entropy':
                train_error_epoch = train_supervised_crossEntropy(net, jparams, train_loader, epoch)

            test_error_epoch = test_supervised_ep(net, test_loader)

            #train_error_list.append(train_error.cpu().item())
            train_error_list.append(train_error_epoch.item())
            test_error_list.append(test_error_epoch.item())

            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, train_error_list, test_error_list)
            # save the inference model
            #t orch.save(net.state_dict(), BASE_PATH)
            # save the entire model
            with open(BASE_PATH + prefix + 'model_entire.pt', 'wb') as f:
                torch.jit.save(net, f)
            # torch.save(net, BASE_PATH + prefix + 'model_entire.pt')

    elif jparams['action'] == 'unsupervised_ep':
        print("Training the model with unsupervised ep")

        #saveHyperparameters(args, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, method='unsupervised')

        # dataframe for Xth
        Xth_dataframe = initXthframe(BASE_PATH, 'Xth_norm.csv')

        # save the initial network
        with open(BASE_PATH + prefix + 'model_initial.pt', 'wb') as f:
            torch.jit.save(net, f)

        #net.save(BASE_PATH + prefix+'model_entire.pt')

        # create a forward NN with same weights of EP network
        # forward_net = forwardNN(jparams['fcLayers'], net.W, jparams['Prune'], jparams['PruneAmount'])

        test_error_list_av = []
        test_error_list_max = []

        Xth_record = []
        Max0_record = []

        for epoch in tqdm(range(jparams['epochs'])):

            # train process
            if jparams['lossFunction'] == 'MSE':
                Xth = train_unsupervised_ep(net, jparams, train_loader, epoch)
            elif jparams['lossFunction'] == 'Cross-entropy':
                Xth = train_unsupervised_crossEntropy(net, jparams, train_loader, epoch)

            # elif args.Optimizer == 'Adam':
            #     Xth, mW, vW, mBias, vBias = train_unsupervised_ep(net, args, train_loader, epoch, Xth, mW, vW, mBias, vBias)

            # one2one class process
            response, max0_indice = classify(net, jparams, class_loader)

            # test process
            error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, test_loader, response)

            test_error_list_av.append(error_av_epoch.item())
            test_error_list_max.append(error_max_epoch.item())

            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME,  test_error_list_av, test_error_list_max)
            # # save the inference model
            # torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict.pt')

            # save the entire model
            with open(BASE_PATH + prefix + 'model_entire.pt', 'wb') as f:
                torch.jit.save(net, f)

            Xth_record.append(torch.norm(Xth).item())
            Xth_dataframe = updateXthframe(BASE_PATH, Xth_dataframe, Xth_record)

        # we create the layer for classfication
        class_net = Classlayer(jparams)

        # we create dataframe for classification layer
        class_dataframe = initDataframe(BASE_PATH, method='classification_layer', dataframe_to_init='classification_layer.csv')
        torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict_0.pt')
        class_train_error_list = []
        final_test_error_list = []
        final_loss_error_list = []

        # at the end we train the final classification layer
        for epoch in tqdm(range(jparams['class_epoch'])):
            # we train the classification layer
            class_train_error_epoch = classify_network(net, class_net, jparams, layer_loader)
            class_train_error_list.append(class_train_error_epoch.item())
            # we test the final test error
            final_test_error_epoch, final_loss_epoch = test_unsupervised_ep_layer(net, class_net, jparams, test_loader)
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = updateDataframe(BASE_PATH, class_dataframe, class_train_error_list, final_test_error_list,
                                              filename='classification_layer.csv', loss=final_loss_error_list)

            # save the trained class_net
            torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict.pt')
            #print('This is epoch:', epoch, 'The 0 responses neuron is:', max0_indice)

    elif jparams['action'] == 'visu':

        # # the hyper-parameters "analysis_preTrain" should be set at 1 at the beginning
        del(jparams)
        # load the json file
        with open(r'D:\Results_data\EP_batch_optimized_results\784-2000-N7-beta0.31-hardsigm-lr0.0159-batch139-gamma0.2-epoch100\config.json') as f:
            jparams = json.load(f)
        jparams['action'] = 'visu'
        # # reverse several variables in jparams
        jparams['fcLayers'].reverse()  # we put in the other side, output first, input last
        jparams['C_list'].reverse()  # we reverse also the list of channels
        jparams['lr'].reverse()
        jparams['display'].reverse()
        jparams['imShape'].reverse()
        jparams['dropProb'].reverse()
        jparams['pruneAmount'].reverse()

        # load the pre-trained network
        with open(r'D:\Results_data\EP_batch_optimized_results\784-2000-N7-beta0.31-hardsigm-lr0.0159-batch139-gamma0.2-epoch100\S-10\model_entire.pt', 'rb') as f:
            loaded_net = torch.jit.load(f)

        net = torch.jit.script(MlpEP(jparams))
        net.W = loaded_net.W.copy()
        net.bias = loaded_net.bias.copy()

        net.eval()
        # reset the dataset
        label_percentage = jparams['label_percentage']
        if jparams['label_percentage'] == 1:
            class_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms))
            layer_set = train_set
        else:
            class_set = splitClass(x, y, label_percentage, seed=seed,
                                   transform=torchvision.transforms.Compose(transforms))
            layer_set = splitClass(x, y, label_percentage, seed=seed,
                                   transform=torchvision.transforms.Compose(transforms),
                                   target_transform=ReshapeTransformTarget(10))
        if jparams['device'] >= 0:
            class_loader = torch.utils.data.DataLoader(class_set, batch_size=jparams['test_batchSize'], shuffle=True)
        else:
            class_loader = torch.utils.data.DataLoader(class_set, batch_size=jparams['test_batchSize'], shuffle=True)

        layer_loader = torch.utils.data.DataLoader(layer_set, batch_size=jparams['test_batchSize'], shuffle=True)

        # one2one response
        response, max0_indice = classify(net, jparams, class_loader)
        error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, test_loader, response)

        # result of one2one
        error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, test_loader, response)
        one2one_result = [error_av_epoch.cpu().numpy(), error_max_epoch.cpu().numpy()]
        print(one2one_result)
        np.savetxt(BASE_PATH + prefix + 'one2one.txt', one2one_result, delimiter=',')

        # we create the classification layer
        class_net = Classlayer(jparams)

        # dataframe save classification layer training result
        class_dataframe = initDataframe(BASE_PATH, method='classification_layer',
                                        dataframe_to_init='classification_layer'+str(label_percentage)+'.csv')
        torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict_0.pt')

        class_train_error_list = []
        final_test_error_list = []
        final_loss_error_list = []

        # at the end we train the final classification layer
        for epoch in tqdm(range(jparams['class_epoch'])):
            # we train the classification layer
            class_train_error_epoch = classify_network(net, class_net, jparams, layer_loader)
            class_train_error_list.append(class_train_error_epoch.item())
            # we test the final test error
            final_test_error_epoch, final_loss_epoch = test_unsupervised_ep_layer(net, class_net, jparams, test_loader)
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = updateDataframe(BASE_PATH, class_dataframe, class_train_error_list, final_test_error_list,
                                              filename='classification_layer'+str(label_percentage)+'.csv', loss=final_loss_error_list)

            # save the trained class_net
            torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict.pt')

        # TODO use the trained class net to give the direct response
        # class_net.load_state_dict(torch.load(r'C:\...'))
        # final_test_error, final_loss = test_unsupervised_ep_layer(net, class_net, args, test_loader)

    if jparams['dataset'] == 'YinYang':
    # TODO visualize the YinYang separation result
        print('we print the classification results for the YinYang dataset')
        # create the visualization dossier
        path_YinYang = pathlib.Path(BASE_PATH + prefix + 'YinYang_visu')
        path_YinYang.mkdir(parents=True, exist_ok=True)

        if jparams['action'] == 'supervised_ep':
            test_error_final, test_class_record = test_supervised_ep(net, test_loader, record=1)
        elif jparams['action'] == 'unsupervised_ep':
            # class process
            response, max0_indice = classify(net, jparams, class_loader)

            # test process
            error_av_final, error_max_epoch_final, av_record, max_record = test_unsupervised_ep(net, jparams, test_loader, response, record=1)

        visu_YinYang_result(av_record.cpu(), test_set.vals, path_YinYang, prefix)
    if jparams['imWeights']:
        # create the imShow dossier
        path_imshow = pathlib.Path(BASE_PATH + prefix + 'imShow')
        path_imshow.mkdir(parents=True, exist_ok=True)
        # for several layers structure
        for i in range(len(jparams['fcLayers']) - 1):
            figName = 'layer' + str(i) + ' weights'
            display = jparams['display'][2 * i:2 * i + 2]
            imShape = jparams['imShape'][2 * i:2 * i + 2]
            weights = net.W[i]
            if jparams['device'] >= 0:
                weights = weights.cpu()
            plot_imshow(weights, jparams['fcLayers'][i], display, imShape, figName, path_imshow, prefix)
            # plot the distribution of weights
            plot_distribution(weights, jparams['fcLayers'][i], 'dis' + figName, path_imshow, prefix)

        # calculate the overlap matrix
        # TODO to verify the overlap
        if len(jparams['fcLayers']) > 2:
            overlap = net.W[-1]
            for j in range(len(jparams['fcLayers']) - 2):
                overlap = torch.mm(overlap, net.W[-j - 2])
            if jparams['device'] >= 0:
                overlap = overlap.cpu()
            display = jparams['display'][0:2]
            imShape = jparams['imShape'][-2:]
            plot_imshow(overlap, jparams['fcLayers'][0], display, imShape, 'overlap', path_imshow, prefix)

    if jparams['maximum_activation']:
        # create the maximum activation dossier
        path_activation = pathlib.Path(BASE_PATH + prefix + 'maxActivation')
        path_activation.mkdir(parents=True, exist_ok=True)

        image_beta_value = 0.5

        if jparams['dataset'] == 'digits':
            data_average = 3.8638
            lr = 0.1
            nb_epoch = 100
        elif jparams['dataset'] == 'mnist':
            data_average = 9.20
            lr = 0.2
            nb_epoch = 500

        # return the responses of output neurons
        response, max0_indice = classify(net, jparams, class_loader)
        # we choose the display form of output neurons
        display = jparams['display'][0:2]
        imShape = jparams['imShape'][-2:]
        neuron_per_class = int(display[0]*display[1]/jparams['n_class'])

        indx_neurons = []

        # select the neuron to be presented at the beginning
        for i in range(jparams['n_class']):
            index_i = (response.cpu() == i).nonzero(as_tuple=True)[0].numpy()
            np.random.shuffle(index_i)

            range_index = min(len(index_i), neuron_per_class)
            indx_neurons.extend(index_i[0:range_index])

        # create a tensor including one image for each selected neuron
        # TODO test this possibility first by bp
        image_max = torch.rand(display[0]*display[1], jparams['fcLayers'][-1], 1,  requires_grad=False, device=net.device)

        for i in range(len(indx_neurons)):
            image = torch.rand(imShape[0], imShape[1], 1, requires_grad=False, device=net.device)
            # SGD process
            for epoch in range(nb_epoch):
                # # there is no need!!
                # optimizer = torch.optim.SGD([image_max], lr=lr)
                # optimizer.zero_grad()
                # TODO depend on the network structure, we decide to do the flatten of image or not
                if jparams['convNet'] == 0:
                    data = image.view(image.size(-1), -1)
                    # TODO we should change the size back to 28*28!
                # initialize
                s = net.initState(data)
                # transfer to cuda
                if net.cuda:
                    s = [item.to(net.device) for item in s]
                # free phase
                s = net.forward(s)
                seq = s.copy()
                # we create the beta and targets
                image_beta = torch.zeros(jparams['fcLayers'][0], device=net.device)
                image_beta[indx_neurons[i]] = image_beta_value
                image_target = torch.zeros(s[0].size(), device=net.device)
                image_target[0, indx_neurons[i]] = 1
                # nudging phase
                # s = net.forward(s, target=image_target, beta=image_beta_value)
                for t in range(jparams['Kmax']):
                    s = net.stepper_c_ep_vector_beta(s, target=image_target, beta=image_beta)
                #s = net.forward(s, target=image_target, beta=image_beta)

                # update the input data
                gradImage = (1/jparams['beta'])*rhop(s[-1])*torch.mm(rho(s[-2])-rho(seq[-2]), net.W[-1].weight)
                data = data + lr*gradImage

                # reshape the input data to the image size
                # TODO to verify whether it is compatible with colorful images
                image = data.view(imShape[0], imShape[1], data.size(0))
                image = data_average * (image.data / torch.norm(image.data[:, 0]).item())
            #     image = (data_average * (image.data / torch.norm(image.data[:, 0]).item())).requires_grad_(True)
            image_max[i, :] = image.view(-1, image.size(-1))

        # plot the figure
        figName = ' max activation figures of output neurons'
        plot_imshow(image_max.cpu(), display[0]*display[1], display, imShape, figName, path_activation, prefix)


            # # This part does not apply the batch
            # for j in range(len(args.fcLayers)-1):
            #     # image_max = torch.zeros(display[0]*display[1])
            #     image_max = torch.zeros(args.fcLayers[-j], args.fcLayers[0])
            #
            #     for i in range(args.fcLayers[j]):
            #
            #         #all_responses, total_unclassified = classify_layers(net, args, class_loader)
            #
            #         # batch version
            #         image = torch.rand((args.fcLayers[j], args.fcLayers[0]), 1, requires_grad=True, device=net.device)
            #         # This process can cost lots of time!!!
            #
            #         # Chose the neuron to be presented at the beginning
            #
            #         for epoch in range(nb_epoch):
            #
            #             optimizer = torch.optim.SGD([image], lr=lr)
            #             optimizer.zero_grad()
            #             output = image
            #             for k in range(j+1):
            #                 # if k == j:
            #                 #     rho = 'x'
            #                 # else:
            #                 rho = args.rho[k]
            #                 # forward propagation
            #
            #                 output = net.rho(torch.mm(net.W[k].weight, output), rho)
            #             # print(output)
            #             # The loss should be a scalar
            #             loss = -output[i, 0]
            #             loss.backward()
            #             optimizer.step()
            #             image = (data_average * (image.data / torch.norm(image.data[:, 0]).item())).requires_grad_(True)
            #         image_max[i, :] = image[:, 0].detach().cpu()
            #     figName = 'layer' + str(j) + ' max activation figures'
            #     imShape = args.imShape[-2:]
            #     display = args.display[2 * j:2 * j + 2]
            #     plot_imshow(image_max, args.layersList[j+1], display, imShape, figName, path_activation, prefix)


# win: cd Desktop/Code/EP-BENCHMARK
#mac: cd Desktop/Thèse/Code/EP-BENCHMARK
#to run: cf. README in the same folder

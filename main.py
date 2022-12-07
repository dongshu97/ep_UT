#Main for the simulation
import os
import argparse
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

parser = argparse.ArgumentParser(description='usupervised EP')
parser.add_argument(
    '--device',
    type=int,
    default=0,
    help='GPU name to use cuda')
parser.add_argument(
    '--dataset',
    type=str,
    default="YinYang",
    help='dataset to be used to train the network : (default = mnist)')
parser.add_argument(
    '--action',
    type=str,
    default="supervised_ep",
    help='train or test: (default = unsupervised_ep, other: supervised_ep, test, visu')
parser.add_argument(
    '--epochs',
    type=int,
    default=50,
    metavar='N',
    help='number of epochs to train (default: 100)')
parser.add_argument(
    '--batchSize',
    type=int,
    default=20,
    help='Batch size (default=128)')
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
    '--T',
    type=int,
    default=30,
    help='number of time steps in the free phase (default: 40) - Let the system relax with oscillators dynamics')
parser.add_argument(
    '--Kmax',
    type=int,
    default=10,
    help='number of time steps in the backward pass (default: 50)')
parser.add_argument(
    '--beta',
    type=float,
    default=0.5,
    metavar='BETA',
    help='nudging parameter (default: 0.5)')
parser.add_argument(
    '--clamped',
    type=int,
    default=1,
    help='Clamped state of the network: crossed input are clamped to avoid divergence (default: True)')
parser.add_argument(
    '--convNet',
    type=int,
    default=0,
    help='whether use the convolutional layers'
)
parser.add_argument(
    '--convLayers',
    nargs='+',
    type=int,
    default=[1, 16, 5, 1, 1, 16, 32, 5, 1, 1],
    help='The parameters of convNet, each conv layer has 5 parameter: in_channels, out_channels, K/F, S, P')
parser.add_argument(
    '--fcLayers',
    nargs='+',
    type=int,
    default=[4, 10, 3],
    help='The parameters of convNet, each conv layer has 5 parameter: in_channels, out_channels, K/F, S, P')
parser.add_argument(
    '--lr',
    nargs='+',
    type=float,
    default=[0.01, 0.01],
    help='learning rates')
parser.add_argument(
    '--activation_function',
    type=str,
    default='hardsigm',
    help='activation function')
parser.add_argument(
    '--eta',
    type=float,
    default=0.2,
    help='the coefficient for regulating the homeostasis effect (default: 0.1)'
)
parser.add_argument(
    '--gamma',
    type=float,
    default=0.518,
    help='the coefficient for regulating the homeostasis effect (default: 0.2)'
)
parser.add_argument(
    '--nudge_N',
    type=int,
    default=1,
    help='the number of winners to be nudged (default: 1)'
)
parser.add_argument(
    '--n_class',
    type=int,
    default=10,
    help='the number of class (default = 10)'
)
parser.add_argument(
    '--Optimizer',
    type=str,
    default='SGD',
    help='the optimizer to be used (default=SGD, else:Adam)'
)
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
    default=1,
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
    default=0,
    help='to decide whether to use the Dropout'
)
parser.add_argument(
    '--dropProb',
    nargs='+',
    type=float,
    default=[0.2, 0.4],
    help='to decide the probability of dropout'
)
parser.add_argument(
    '--analysis_preTrain',
    type=str,
    default=0,
    help='whether to load the trained model'
)
parser.add_argument(
    '--imWeights',
    type=int,
    default=0,
    help='whether we imshow the weights of synapses'
)
parser.add_argument(
    '--maximum_activation',
    type=int,
    default=0,
    help='draw the maximum activation input for each neuron'
)
parser.add_argument(
    '--imShape',
    nargs='+',
    type=int,
    default=[28, 28, 32, 32],
    help='decide the size for each imshow of weights'
)
parser.add_argument(
    '--display',
    nargs='+',
    type=int,
    default=[10, 10, 10, 10],
    help='decide the number of neurons whose weights are presented'
)
# input the args
args = parser.parse_args()

# define the two batch sizes
batch_size = args.batchSize
batch_size_test = args.test_batchSize

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


if args.dataset == 'mnist':
    print('We use the MNIST Dataset')
    # Define the Transform
    if args.convNet:
        transforms = [torchvision.transforms.ToTensor()]
    else:
        transforms = [torchvision.transforms.ToTensor(), ReshapeTransform((-1,))]

    # Download the MNIST dataset
    if args.action == 'unsupervised_ep' or args.action == 'unsupervised_conv_ep' or args.action=='test':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms))
        # # we create the dataset mixed with unlabeled and labeled data
        # if args.unlabeledPercent != 0 and args.unlabeledPercent != 1:
        #     train_set = UnlabelDataset(train_set, './MNIST_alterLearning', args.unlabeledPercent, Seed=0)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchSize, shuffle=True)

    elif args.action == 'supervised_ep' or args.action == 'train_conv_ep':
        train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.Compose(transforms),
                                               target_transform=ReshapeTransformTarget(10))
        # if args.unlabeledPercent != 0 and args.unlabeledPercent != 1:
        #     train_set = subMNISTDataset(root='./MNIST_alterLearning', train_set=train_set,
        #                                 unlabeledPercent=args.unlabeledPercent,
        #                                 Seed=0, transform=torchvision.transforms.Compose(transforms),
        #                                 target_transform=ReshapeTransformTarget(10))
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchSize, shuffle=True)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                          transform=torchvision.transforms.Compose(transforms))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batchSize, shuffle=True)


    # define the class dataset
    seed = 1

    # TODO this part can use the same method as data-split of semi-supervised learning
    # or TODO this separation by torch.utils.data.Subset(dataset, indices)
    class_set = ClassDataset(root='./MNIST_class_seed', test_set=test_set, seed=seed,
                             transform=torchvision.transforms.Compose(transforms))

    if args.device >= 0:
        class_loader = torch.utils.data.DataLoader(class_set, batch_size=1000, shuffle=True)
    else:
        class_loader = torch.utils.data.DataLoader(class_set, batch_size=args.test_batchSize, shuffle=True)

elif args.dataset == 'YinYang':
    print('We use the YinYang dataset')
    ''' This dataset is not considered to apply with a ConvNet'''
    if args.convNet:
        raise ValueError("YinYang dataset is not designed for ConvNet, the value should not be " "but got {}".format(args.convNet))
    else:
        if args.action == 'supervised_ep':
            train_set = YinYangDataset(size=5000, seed=42, target_transform=ReshapeTransformTarget(3))
        elif args.action == 'unsupervised_ep' or args.action == 'test' or args.action=='visu':
            train_set = YinYangDataset(size=5000, seed=42)

        #validation_set = YinYangDataset(size=1000, seed=41)  # used for the hyperparameter research

        test_set = YinYangDataset(size=1000, seed=40)

        class_set = YinYangDataset(size=1000, seed=40, sub_class=True)

        # seperate the dataset
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batchSize, shuffle=True)
        #validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batchSize, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batchSize, shuffle=False)
        class_loader = torch.utils.data.DataLoader(class_set, batch_size=100, shuffle=False)


#  TODO finish the CIFAR 10 dataset
elif args.dataset == 'CIFAR10':
    print('We use the CIFAR10 Dataset')
    # TODO to complete the CIFAR10


# define the activation function
if args.activation_function == 'sigm':
    def rho(x):
        return 1/(1+torch.exp(-(4*(x-0.5))))

    def rhop(x):
        return 4*torch.mul(rho(x), 1 -rho(x))

elif args.activation_function == 'hardsigm':
    def rho(x):
        return x.clamp(min=0).clamp(max = 1)

    def rhop(x):
        return (x >= 0) & (x <= 1)

elif args.activation_function == 'tanh':
    def rho(x):
        return torch.tanh(x)

    def rhop(x):
        return 1 - torch.tanh(x)**2

elif args.activation_function == 'relu':
    def rho(x):
        return x.clamp(min=0)

    def rhop(x):
        return (x>=0)


if __name__ == '__main__':

    args.fcLayers.reverse()  # we put in the other side, output first, input last
    args.lr.reverse()
    args.display.reverse()
    args.imShape.reverse()
    args.dropProb.reverse()

    BASE_PATH, name = createPath(args)

    # we create the network
    if args.convNet:
        net = ConvEP(args)
    else:
        net = torch.jit.script(MlpEP(args))

    # we load the pre-trained network
    if args.analysis_preTrain:
        with open(r'C:/model_entire.pt', 'rb') as f:
            net = torch.jit.load(f)
        net.eval()

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if args.action == 'test':
        print("Testing the model")

        saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, args, net)
        print(DATAFRAME)

        data, target = next(iter(train_loader))

        s = net.initHidden(args, data)
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
        net.updateWeight(s, seq, args.beta)

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
        plt.axvline(args.T-1, ymin=0, ymax=0.8, ls='--', linewidth=1, color='b')
        plt.axvline(args.T + args.Kmax-1, ymin=0, ymax=0.8, ls='--', linewidth=1, color='b')
        plt.title('Dynamics of Equilibrium Propagation')
        fig.savefig(BASE_PATH + prefix + 'dymanics.png', format='png', dpi=300)
        plt.show()

    elif args.action == 'supervised_ep':
        print("Training the model with supervised ep")

        saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, args, net, method='supervised')

        # save the initial network
        torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict_0.pt')

        train_error_list = []
        test_error_list = []

        for epoch in tqdm(range(args.epochs)):

            train_error_epoch = train_supervised_ep(net, args, train_loader, epoch)

            test_error_epoch = test_supervised_ep(net, args, test_loader)

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

    elif args.action == 'unsupervised_ep':
        print("Training the model with unsupervised ep")

        saveHyperparameters(args, net, BASE_PATH)
        DATAFRAME = initDataframe(BASE_PATH, args, net, method='unsupervised')

        # dataframe for Xth
        Xth_dataframe = initXthframe(BASE_PATH, 'Xth_norm.csv')


        # save the initial network
        torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict_0.pt')

        #net.save(BASE_PATH + prefix+'model_entire.pt')

        test_error_list_av = []
        test_error_list_max = []

        Xth_record = []
        Max0_record = []

        for epoch in tqdm(range(args.epochs)):

            # train process
            Xth = train_unsupervised_ep(net, args, train_loader, epoch)

            # elif args.Optimizer == 'Adam':
            #     Xth, mW, vW, mBias, vBias = train_unsupervised_ep(net, args, train_loader, epoch, Xth, mW, vW, mBias, vBias)

            # class process
            response, max0_indice = classify(net, args, class_loader)

            # test process
            error_av_epoch, error_max_epoch = test_unsupervised_ep(net, args, test_loader, response)

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

            #print('This is epoch:', epoch, 'The 0 responses neuron is:', max0_indice)

    elif args.action == 'visu':
        # TODO write the visu version for the

        # the hyper-parameters "analysis_preTrain" should be set at 1 at the beginning
        module_W = nn.ModuleList(None)

        with torch.no_grad():
            for i in range(len(args.fcLayers)-1):
                module_W.extend([nn.Linear(args.fcLayers[i+1], args.fcLayers[i], bias=True)])

        with open(r'C:\Users\CNRS-THALES\OneDrive\文档\Homeostasis_python\Eqprop-unsuperivsed-MLP\DATA-0\2022-12-01\S-6\model_entire.pt', 'rb') as f:
            net = torch.jit.load(f)
        net.eval()

        response, max0_indice = classify(net, args, class_loader)
        error_av_epoch, error_max_epoch = test_unsupervised_ep(net, args, test_loader, response)

    if args.imWeights:
        # create the imShow dossier
        path_imshow = pathlib.Path(BASE_PATH + prefix + 'imShow')
        path_imshow.mkdir(parents=True, exist_ok=True)
        # for several layers structure
        for i in range(len(args.fcLayers) - 1):
            figName = 'layer' + str(i) + ' weights'
            display = args.display[2 * i:2 * i + 2]
            imShape = args.imShape[2 * i:2 * i + 2]
            weights = net.W[i]
            if args.device >= 0:
                weights = weights.cpu()
            plot_imshow(weights, args.fcLayers[i], display, imShape, figName, path_imshow, prefix)
            # plot the distribution of weights
            plot_distribution(weights, args.fcLayers[i], 'dis' + figName, path_imshow, prefix)

        # calculate the overlap matrix
        # TODO to verify the overlap
        if len(args.fcLayers) > 2:
            overlap = net.W[-1]
            for j in range(len(args.fcLayers) - 2):
                overlap = torch.mm(overlap, net.W[-j - 2])
            if args.device >= 0:
                overlap = overlap.cpu()
            display = args.display[0:2]
            imShape = args.imShape[-2:]
            plot_imshow(overlap, args.fcLayers[0], display, imShape, 'overlap', path_imshow, prefix)

    if args.maximum_activation:
        # create the maximum activation dossier
        path_activation = pathlib.Path(BASE_PATH + prefix + 'maxActivation')
        path_activation.mkdir(parents=True, exist_ok=True)

        image_beta_value = 0.5

        if args.dataset == 'digits':
            data_average = 3.8638
            lr = 0.1
            nb_epoch = 100
        elif args.dataset == 'mnist':
            data_average = 9.20
            lr = 0.2
            nb_epoch = 500

        # return the responses of output neurons
        response, max0_indice = classify(net, args, class_loader)
        # we choose the display form of output neurons
        display = args.display[0:2]
        imShape = args.imShape[-2:]
        neuron_per_class = int(display[0]*display[1]/args.n_class)

        indx_neurons = []

        # select the neuron to be presented at the beginning
        for i in range(args.n_class):
            index_i = (response.cpu() == i).nonzero(as_tuple=True)[0].numpy()
            np.random.shuffle(index_i)

            range_index = min(len(index_i), neuron_per_class)
            indx_neurons.extend(index_i[0:range_index])

        # create a tensor including one image for each selected neuron
        # TODO test this possibility first by bp
        image_max = torch.rand(display[0]*display[1], args.fcLayers[-1], 1,  requires_grad=False, device=net.device)

        for i in range(len(indx_neurons)):
            image = torch.rand(imShape[0], imShape[1], 1, requires_grad=False, device=net.device)
            # SGD process
            for epoch in range(nb_epoch):
                # # there is no need!!
                # optimizer = torch.optim.SGD([image_max], lr=lr)
                # optimizer.zero_grad()
                # TODO depend on the network structure, we decide to do the flatten of image or not
                if args.convNet == 0:
                    data = image.view(image.size(-1), -1)
                    # TODO we should change the size back to 28*28!
                # initialize
                s = net.initHidden(args, data)
                # transfer to cuda
                if net.cuda:
                    s = [item.to(net.device) for item in s]
                # free phase
                s = net.forward(s)
                seq = s.copy()
                # we create the beta and targets
                image_beta = torch.zeros(args.fcLayers[0], device=net.device)
                image_beta[indx_neurons[i]] = image_beta_value
                image_target = torch.zeros(s[0].size(), device=net.device)
                image_target[0, indx_neurons[i]] = 1
                # nudging phase
                # s = net.forward(s, target=image_target, beta=image_beta_value)
                for t in range(args.Kmax):
                    s = net.stepper_c_ep_vector_beta(s, target=image_target, beta=image_beta)
                #s = net.forward(s, target=image_target, beta=image_beta)

                # update the input data
                gradImage = (1/args.beta)*rhop(s[-1])*torch.mm(rho(s[-2])-rho(seq[-2]), net.W[-1].weight)
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

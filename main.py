#Main for the simulation
import os
import argparse  # this can be removed
import json
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import datetime
import numpy as np
import pathlib

from Data import *
from actions import *
from Tools import *
from Network import *
from plotFunction import *
from visu import *

# TODO use argsparser to give the file path of json file
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
    # TODO change the default path back
    default=r'D:\Eqprop-unsuperivsed-MLP\unsupervised_results\2layer\lr0.01-0.015_beta0.2_batch32_drop0.2_0_0.3_decay1e-3\S-1',
    help='path of model_dict_state_file'
)

args = parser.parse_args()

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


if jparams['dataset'] == 'mnist':
    print('We use the MNIST Dataset')
    (train_loader, test_loader,
     class_loader, layer_loader,
     supervised_loader, unsupervised_loader) = returnMNIST(jparams, validation=False)

elif jparams['dataset'] == 'cifar10':
    print('We use the CIFAR10 dataset')
    (train_loader, test_loader,
     class_loader, layer_loader,
     supervised_loader, unsupervised_loader) = returnCifar10(jparams, validation=False)

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

elif jparams['activation_function'] == 'half_hardsigm':
    def rho(x):
        return (1 + F.hardtanh(x - 1))*0.5
    def rhop(x):
        return ((x >= 0) & (x <= 2))*0.5

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
    jparams['C_list'].reverse() # we reverse also the list of channels
    jparams['Pad'].reverse()
    jparams['lr'].reverse()
    jparams['pre_lr'].reverse()
    jparams['display'].reverse()
    jparams['imShape'].reverse()
    jparams['dropProb'].reverse()
    jparams['pruneAmount'].reverse()

    BASE_PATH, name = createPath()
    # save hyper-parameters as json file
    with open(BASE_PATH + prefix + "config.json", "w") as outfile:
        json.dump(jparams, outfile)

    # Cuda problem
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    # we create the network and define the  parameters
    if jparams['convNet']:
        net = ConvEP(jparams, rho, rhop)
    else:
        net = torch.jit.script(MlpEP(jparams, rho, rhop))

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
        supervised_ep(net, jparams, train_loader, test_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'unsupervised_ep':
        print("Training the model with unsupervised ep")
        unsupervised_ep(net, jparams, train_loader, class_loader, test_loader, layer_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'train_class_layer':
        print("Train the supplementary class layer for unsupervised learning")
        trained_path = args.trained_path + prefix + 'model_entire.pt'
        train_class_layer(net, jparams, layer_loader, test_loader, trained_path=trained_path, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'semi-supervised_ep':
        print("Training the model with semi-supervised learning")
        semi_supervised_ep(net, jparams, supervised_loader, unsupervised_loader, test_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'pre_train_ep':
        print("Training the model with little dataset with supervised ep")
        pre_supervised_ep(net, jparams, supervised_loader, test_loader, BASE_PATH=BASE_PATH)

    elif jparams['action'] == 'visu':
        # load the pre-trained network
        if jparams['convNet']:
            net = ConvEP(jparams, rho, rhop)
            net.load_state_dict(torch.load(args.trained_path + prefix +'model_state_dict_entire.pt'))
        else:
            with open(args.trained_path + prefix + 'model_entire.pt', 'rb') as f:
                loaded_net = torch.jit.load(f, map_location=net.device)
                net = torch.jit.script(MlpEP(jparams, rho, rhop))
                net.W = loaded_net.W.copy()
                net.bias = loaded_net.bias.copy()

        net.eval()

        # create dataframe
        DATAFRAME = initDataframe(BASE_PATH, method='unsupervised')
        test_error_list_av = []
        test_error_list_max = []
        # one2one
        response = classify(net, jparams, class_loader)
        # test process
        error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, test_loader, response)
        test_error_list_av.append(error_av_epoch.item())
        test_error_list_max.append(error_max_epoch.item())
        DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, test_error_list_av, test_error_list_max)

        # we create the layer for classfication
        train_class_layer(net, jparams, layer_loader, test_loader, trained_path=None, BASE_PATH=BASE_PATH, trial=None)


        # # we create the classification layer
        # class_net = Classlayer(jparams)
        #
        # # dataframe save classification layer training result
        # class_dataframe = initDataframe(BASE_PATH, method='classification_layer',
        #                                 dataframe_to_init='classification_layer'+str(classLabel_percentage)+'.csv')
        # torch.save(class_net.state_dict(), os.path.join(BASE_PATH, 'class_model_state_dict_0.pt'))
        #
        # class_train_error_list = []
        # final_test_error_list = []
        # final_loss_error_list = []

        # at the end we train the final classification layer
        # for epoch in tqdm(range(jparams['class_epoch'])):
        #     # we train the classification layer
        #     class_train_error_epoch = classify_network(net, class_net, jparams, layer_loader)
        #     class_train_error_list.append(class_train_error_epoch.item())
        #     # we test the final test error
        #     final_test_error_epoch, final_loss_epoch = test_unsupervised_ep_layer(net, class_net, jparams, test_loader)
        #     final_test_error_list.append(final_test_error_epoch.item())
        #     final_loss_error_list.append(final_loss_epoch.item())
        #     class_dataframe = updateDataframe(BASE_PATH, class_dataframe, class_train_error_list, final_test_error_list,
        #                                       filename='classification_layer'+str(classLabel_percentage)+'.csv', loss=final_loss_error_list)
        #
        #     # save the trained class_net
        #     torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict.pt')

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
            test_error_final, test_class_record = test_supervised_ep(net, jparams, test_loader, jparams['lossFunction'], record=1)
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

    # TODO do the maximum activation by reverse propagation
    if jparams['reverseProp']:
        # create the reconstructed images at the beginning
        path_activation = pathlib.Path(BASE_PATH + prefix + 'reverseProp')
        path_activation.mkdir(parents=True, exist_ok=True)

        # return the responses of output neurons
        response, max0_indice = classify(net, jparams, class_loader)
        # we choose the display form of output neurons
        display = jparams['display'][-2:]
        imShape = jparams['imShape'][-2:]
        neuron_per_class = int(display[0] * display[1] / jparams['n_class'])

        indx_neurons = []
        indx_all_class = []
        max_neurons_per_class = jparams['fcLayers'][0]

        # select the neuron to be presented at the beginning
        for i in range(jparams['n_class']):
            index_i = (response.cpu() == i).nonzero(as_tuple=True)[0].numpy()
            np.random.shuffle(index_i)
            indx_all_class.append(index_i)

            max_neurons_per_class = min(max_neurons_per_class, len(index_i))
            range_index = min(len(index_i), neuron_per_class)
            indx_neurons.extend(index_i[0:range_index])

        indx_all_class_torch = torch.zeros([10, max_neurons_per_class], dtype=torch.int64, device=net.device)
        for i in range(jparams['n_class']):
            indx_all_class_torch[i, :] = torch.tensor(indx_all_class[i][0:max_neurons_per_class])

        # create a tensor including one image for each selected neuron
        # TODO make a compatible version for CNN

        # generate the target
        nudge_target = torch.zeros((len(indx_neurons), jparams['fcLayers'][0]), requires_grad=False, device=net.device)
        indx_neurons = torch.tensor(indx_neurons).to(net.device).reshape(-1, 1)
        nudge_target.scatter_(1, indx_neurons, torch.ones((len(indx_neurons), jparams['fcLayers'][0]), requires_grad=False, device=net.device))

        # initialize
        clamped_input = torch.zeros((len(indx_neurons), jparams['fcLayers'][-1]), requires_grad=False, device=net.device)

        s = net.generate_image(clamp_input=clamped_input, target=nudge_target)

        # plot the figure
        figName = 'back-propagated input of fixed output values'
        plot_imshow(s[-1].T.cpu(), len(indx_neurons), display, imShape, figName, path_activation, prefix)

        # generate the target for mixed images
        mix_target = torch.zeros((50, jparams['fcLayers'][0]), requires_grad=False, device=net.device)
        for i in range(int(jparams['n_class']/2)):
            for j in range(jparams['n_class']):
                mix_target[(i*jparams['n_class']+j), i] += 0.5
                mix_target[(i * jparams['n_class'] + j), j] += 0.5

        # initialize
        clamped_input = torch.zeros((50, jparams['fcLayers'][-1]), requires_grad=False,
                                    device=net.device)

        s = net.generate_image(clamp_input=clamped_input, target=mix_target)

        # plot the figure
        plot_imshow(s[-1].T.cpu(), 50, [5, 10], imShape, 'generated_images', path_activation, prefix)

        # generate the target for each class
        nudge_class_target = torch.zeros((10, jparams['fcLayers'][0]), requires_grad=False, device=net.device)
        nudge_class_target.scatter_(1, indx_all_class_torch, torch.ones((10, jparams['fcLayers'][0]), requires_grad=False, device=net.device))
        nudge_class_target = nudge_class_target * 1

        # initialize
        clamped_input = torch.zeros([10, jparams['fcLayers'][-1]], requires_grad=False,
                                    device=net.device)
        s = net.generate_image(clamp_input=clamped_input, target=nudge_class_target)

        # plot the figure
        figName = 'back-propagated input of specific class target'
        plot_imshow(s[-1].T.cpu(), 10, [2, 5], imShape, figName, path_activation, prefix)

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

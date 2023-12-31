# File defining the network and the oscillators composing the network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
import torch.nn.utils.prune as prune
from activations import *
from typing import List, Optional, Tuple


'''
Try to use the same function name for the MLP class and Conv class
but the problem is that, there are different variables for Conv class than the MLP class,
so whether we will merge s and h together?
'''

def mydropout(s:List[torch.Tensor], p:List[float], y:Optional[torch.Tensor]=None)->List[torch.FloatTensor]:
    # TODO change it to be a class
    # if p < 0 or p > 1:
    #     raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
    p_distribut = []
    # bernoulli = torch.distributions.bernoulli.Bernoulli(total_count=1, probs=1-p)
    if y is None:
        for layer in range(len(s)):
            if p[layer] == 0:
                p_distribut.append(torch.ones(s[layer].size()))
            else:
                binomial = torch.distributions.binomial.Binomial(probs=torch.tensor(1-p[layer]))
                p_distribut.append(binomial.sample(s[layer].size()))
        return p_distribut
    else:
        if p[0] == 0:
            y_distribut = torch.ones(y.size())
        else:
            binomial = torch.distributions.binomial.Binomial(probs=torch.tensor(1 - p[0]))
            y_distribut = binomial.sample(y.size())
        for layer in range(len(s)):
            if p[layer+1] == 0:
                p_distribut.append(torch.ones(s[layer].size()))
            else:
                binomial = torch.distributions.binomial.Binomial(probs=torch.tensor(1 - p[layer+1]))
                p_distribut.append(binomial.sample(s[layer].size()))
        return p_distribut, y_distribut


def smoothLabels(labels, smooth_factor, nudge_N):
    assert len(labels.shape) == 2, 'input should be a batch of one-hot-encoded data'
    assert 0 <= smooth_factor <= 1, 'smooth_factor should be between 0 and 1'

    if 0 <= smooth_factor <= 1:
        # label smoothing
        labels *= 1 - smooth_factor
        labels += (nudge_N*smooth_factor) / labels.shape[1]
    else:
        raise ValueError('Invalid label smoothing factor: ' + str(smooth_factor))
    return labels


def define_unsupervised_target(output, N, device, Xth=None):

    # define unsupervised target
    unsupervised_targets = torch.zeros(output.size(), device=device)

    # N_maxindex
    if Xth != None:
        N_maxindex = torch.topk(output.detach()-Xth, N).indices  # N_maxindex has the size of (batch, N)
    else:
        N_maxindex = torch.topk(output.detach(), N).indices

    # # !!! Attention if we consider the 0 and 1, which means we expect the values of input is between 0 and 1
    # # This probability occurs when we clamp the 'vector' between 0 and 1
    #
    # # Extract the batch where all the outputs are 0
    # # Consider the case where output is all 0 or 1
    # sum_output = torch.sum(m_output, axis=1)
    # indices_0 = (sum_output == 0).nonzero(as_tuple=True)[0]
    #
    # if indices_0.nelement() != 0:
    #     for i in range(indices_0.nelement()):
    #         N_maxindex[indices_0[i], :] = torch.randint(0, self.output-1, (N,))

    # unsupervised_targets[N_maxindex] = 1
    unsupervised_targets.scatter_(1, N_maxindex, torch.ones(output.size(), device=device))
    # print('the unsupervised vector is:', unsupervised_targets)

    return unsupervised_targets, N_maxindex




class forwardNN(nn.Module):
    def __init__(self, fcLayers, ep_W):
        super(forwardNN, self).__init__()
        # TODO this network only works for FCL
        self.fcLayers = fcLayers
        self.W = nn.ModuleList(None)
        for i in range(len(fcLayers) - 1):
            self.W.extend([nn.Linear(fcLayers[i + 1], fcLayers[i], bias=False)])
            self.W[i].weight.data = ep_W[i]
        # if Prune == 'Initiation':
        #     for i in range(len(pruneAmount)):
        #         # TODO reverse the pruneAmount when we get the hyperparameters
        #         prune.random_unstructured(self.W[i], name='weight', amount=pruneAmount[i])


class MlpEP(jit.ScriptModule):

    def __init__(self, jparams, rho, rhop):

        super(MlpEP, self).__init__()

        self.T = jparams['T']
        self.Kmax = jparams['Kmax']
        self.dt = jparams['dt']
        self.beta = torch.tensor(jparams['beta'])
        self.clamped = jparams['clamped']
        self.batchSize = jparams['batchSize']
        self.fcLayers = jparams['fcLayers']
        self.errorEstimate = jparams['errorEstimate']
        self.gamma = jparams['gamma']
        self.Prune = False
        self.W_mask = [1, 1]
        self.rho = rho
        self.rhop = rhop
        self.nudge_N = jparams['nudge_N']
        # define the device
        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device

        # We define the parameters to be trained

        W:List[torch.Tensor] = []
        for i in range(len(jparams['fcLayers'])-1):
            w = torch.empty(jparams['fcLayers'][i+1], jparams['fcLayers'][i], device=device, requires_grad=True)
            bound = 1 / np.sqrt(jparams['fcLayers'][i+1])
            nn.init.xavier_uniform_(w, gain=0.5)
            #nn.init.uniform_(w, a=-bound, b=bound)
            W.append(w)
        self.W = W

        # We define the list to save the bias
        bias:List[torch.Tensor] = []
        for i in range(len(jparams['fcLayers'])-1):
            b = torch.empty(jparams['fcLayers'][i], device=device, requires_grad=True)
            # bound = 1 / np.sqrt(jparams['fcLayers'][i])
            #nn.init.uniform_(b, a=-bound, b=bound)

            bias.append(b)
        self.bias = bias

        # create the mask for Pruning
        if jparams['Prune'] == 'Initiation':
            self.Prune = True
            if len(jparams['pruneAmount']) != len(jparams['fcLayers'])-1:
                raise ValueError("pruneAmount should be the same size of  network layers")
            forward_W, W_mask = [], []
            for i in range(len(jparams['fcLayers']) - 1):
                forward_W.extend([nn.Linear(jparams['fcLayers'][i + 1], jparams['fcLayers'][i], bias=False)])
                forward_W[i].weight.data = self.W[i]
                prune.random_unstructured(forward_W[i], name='weight', amount=jparams['pruneAmount'][i])
                W_mask.append(forward_W[i].weight_mask)
                self.W[i] = self.W[i].mul(W_mask[i])

            self.W_mask = W_mask

        self = self.to(device)


    @jit.script_method
    # todo combine it with the same stepper
    def stepper_hidden(self, h:List[torch.Tensor], p_distribut:Optional[List[torch.Tensor]], y_distribut:Optional[torch.Tensor],
                       target:Optional[torch.Tensor]=None,
                          beta:Optional[float]=None)->Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:

        y = F.softmax(torch.mm(self.rho(h[0]), self.W[0]) + self.bias[0], dim=1)
        pre_y = torch.mm(self.rho(h[0]), self.W[0]) + self.bias[0]
        if y_distribut is not None:
            y = y_distribut*y
            pre_y = y_distribut * pre_y

        if len(h) > 1:
            dhdt=[]
            dhdt.append(-h[0] + (self.rhop(h[0]) * (torch.mm(self.rho(h[1]), self.W[1]) + self.bias[1])))
            if target is not None and beta is not None:
                dhdt[0] = dhdt[0] + beta * torch.mm((target-y), self.W[0].T)

            for layer in range(1, len(h)-1):
                dhdt.append(-h[layer] + self.rhop(h[layer]) * (
                            torch.mm(self.rho(h[layer + 1]), self.W[layer+1]) + self.bias[layer+1] + torch.mm(self.rho(h[layer - 1]),
                                                                                                     self.W[layer].T)))

            for (layer, dhdt_item) in enumerate(dhdt):
                if p_distribut is not None:
                    # TODO need to rescale the ouput with (1-p)?
                    h[layer] = p_distribut[layer]*(h[layer] + self.dt * dhdt_item)
                else:
                    h[layer] = h[layer] + self.dt * dhdt_item
                if self.clamped:
                    h[layer] = h[layer].clamp(0, 1)

        return h, y, pre_y

    @jit.script_method
    def forward_softmax(self, h:List[torch.Tensor], p_distribut:Optional[List[torch.Tensor]]=None,y_distribut:Optional[torch.Tensor]=None,
                        beta:Optional[float]=None, target:Optional[torch.Tensor]=None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:

        T, Kmax = self.T, self.Kmax

        y = F.softmax(torch.mm(self.rho(h[0]), self.W[0]) + self.bias[0], dim=1)
        pre_y = torch.mm(self.rho(h[0]), self.W[0]) + self.bias[0]

        if y_distribut is not None:
            y = y_distribut*y
            pre_y = y_distribut*pre_y

        with torch.no_grad():
            if beta is None and target is None:
                if len(h) > 1:
                    for t in range(T):
                        h, y, pre_y = self.stepper_hidden(h, p_distribut, y_distribut, target=target, beta=beta)
            else:
                for t in range(Kmax):
                    h, y, pre_y = self.stepper_hidden(h, p_distribut, y_distribut, target=target, beta=beta)
        return h, y, pre_y

    @jit.script_method
    def stepper_c_ep(self, s:List[torch.Tensor], p_distribut: Optional[List[torch.Tensor]], target:Optional[torch.Tensor]=None,
                     beta:Optional[float]=None):
        '''
        stepper function for energy-based dynamics of EP
        '''
        dsdt = []

        dsdt.append(-s[0] + (self.rhop(s[0])*(torch.mm(self.rho(s[1]), self.W[0]) + self.bias[0])))

        if target is not None and beta is not None:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for layer in range(1, len(s)-1):  # start at the first hidden layer and then to the before last hidden layer
            dsdt.append(-s[layer] + self.rhop(s[layer])*(torch.mm(self.rho(s[layer+1]), self.W[layer])+self.bias[layer] + torch.mm(self.rho(s[layer-1]), self.W[layer-1].T)))

        for (layer, dsdt_item) in enumerate(dsdt):
            if p_distribut is not None:
                s[layer] = p_distribut[layer] * (s[layer] + self.dt * dsdt_item)
            else:
                s[layer] = s[layer] + self.dt * dsdt_item
            if self.clamped:
                s[layer] = s[layer].clamp(0, 1)

        return s

    @jit.script_method
    def stepper_generate(self, s:List[torch.Tensor], target:torch.Tensor):
        dsdt = []
        # fix the output
        s[0] = target.clone()
        #dsdt.append(0.5 * (target - s[0]))
        for layer in range(1, len(s) - 1):
            dsdt.append(-s[layer] + self.rhop(s[layer]) * (torch.mm(self.rho(s[layer+1]), self.W[layer]) +
                                                           self.bias[layer] + torch.mm(self.rho(s[layer-1]), self.W[layer-1].T)))

        # for the input layer
        dsdt.append(-s[-1] + (self.rhop(s[-1])) * torch.mm(self.rho(s[-2]), self.W[-1].T)) # no biases

        for (layer, dsdt_item) in enumerate(dsdt):
            s[layer + 1] = s[layer + 1] + self.dt * dsdt_item
            s[layer + 1] = s[layer + 1].clamp(0, 1)

        return s

    def generate_image(self, clamp_input, target):
        s = self.initState(clamp_input)
        # transfer to cuda
        if self.cuda:
            s = [item.to(self.device) for item in s]
        with torch.no_grad():
            for t in range(self.T):
                s = self.stepper_generate(s, target)
        return s

    @jit.script_method
    def forward(self, s:List[torch.Tensor], p_distribut:Optional[List[torch.Tensor]]=None, beta:Optional[float]=None, target:Optional[torch.Tensor]=None,
                ) -> List[torch.Tensor]:

        T, Kmax = self.T, self.Kmax

        with torch.no_grad():
            # continuous time EP
            if beta is None and target is None:
                # free phase
                # TODO this if selection can be improved by giving the T/Kmax outside
                for t in range(T):
                    s = self.stepper_c_ep(s, p_distribut, target=target, beta=beta)
            else:
                # nudged phase
                for t in range(Kmax):
                    s = self.stepper_c_ep(s, p_distribut, target=target, beta=beta)

        return s


    #@jit.script_method
    def computeGradientsEP(self, s:List[torch.Tensor], seq:List[torch.Tensor]):
        '''
        Compute EQ gradient to update the synaptic weight -
        for classic EP! for continuous time dynamics and prototypical
        '''
        batch_size = s[0].size(0)
        # learning rate should be the 1/beta of the BP learning rate
        # in this way the learning rate is correspended with the sign of beta
        coef = 1/(self.beta*batch_size)
        if self.errorEstimate == 'symmetric':
            coef = coef*0.5

        gradW, gradBias = [], []

        with torch.no_grad():
            for layer in range(len(s)-1):
                gradW.append(coef*(torch.mm(torch.transpose(self.rho(s[layer+1]), 0, 1), self.rho(s[layer]))
                                   -torch.mm(torch.transpose(self.rho(seq[layer+1]), 0, 1), self.rho(seq[layer]))))
                gradBias.append(coef*(self.rho(s[layer])-self.rho(seq[layer])).sum(0))

        for i in range(len(self.W)):
            self.W[i].grad = -gradW[i]
            self.bias[i].grad = -gradBias[i]

    #@jit.script_method
    def computeGradientEP_softmax(self, h:List[torch.Tensor], heq:List[torch.Tensor], y:torch.Tensor, target:torch.Tensor,
                                  ybeta:Optional[torch.Tensor]=None):
        # define the coefficient for the hidden neurons
        batch_size = h[0].size(0)
        coef = 1 / (self.beta * batch_size)
        if self.errorEstimate == 'symmetric':
            coef = coef * 0.5
        gradW, gradBias = [], []
        # TODO update the gradients in self.W[i].grad
        with torch.no_grad():
            if ybeta is None:
                gradW.append(-(1/batch_size)*torch.mm(torch.transpose(self.rho(h[0]), 0, 1), (y-target)))
                gradBias.append(-(1/batch_size)*(y-target).sum(0))
            else:
                gradW.append(-(0.5/batch_size)*(torch.mm(torch.transpose(self.rho(h[0]), 0, 1), (y-target)) +
                                          torch.mm(torch.transpose(self.rho(heq[0]), 0, 1), (ybeta-target))))
                gradBias.append(-(0.5/batch_size)*(y+ybeta-2*target).sum(0))
            for layer in range(len(h)-1):
                gradW.append(coef * (torch.mm(torch.transpose(self.rho(h[layer + 1]), 0, 1), self.rho(h[layer]))
                                     - torch.mm(torch.transpose(self.rho(heq[layer + 1]), 0, 1), self.rho(heq[layer]))))
                gradBias.append(coef * (self.rho(h[layer]) - self.rho(heq[layer])).sum(0))
        for i in range(len(self.W)):
            self.W[i].grad = -gradW[i]
            self.bias[i].grad = -gradBias[i]

    @jit.script_method
    def weightNormalization(self):
        '''
        Normalize the last layer of network
        '''
        norm = torch.norm(self.W[0], dim=0)
        scale = torch.mean(norm)
        # scale = 1./torch.sqrt(torch.tensor(self.fcLayers[1]))
        # we normalize the weight by the average norm
        self.W[0] = (self.W[0]/norm)*scale
    #
    # @jit.script_method
    # def inferenceWeight(self, dropProb:List[float]):
    #     for i in range(len(dropProb)-1):
    #         self.W[i] = self.W[i]*dropProb[i+1]
    #
    # @jit.script_method
    # def recoverWeight(self, dropProb:List[float]):
    #     for i in range(len(dropProb)-1):
    #         self.W[i] = self.W[i]/dropProb[i+1]


        # # we devide the weight by the maxmium possible value
        # max_value = torch.max(self.W[0])
        # if max_value > 0:
        #  self.W[0] = self.W[0]/max_value

    @jit.script_method
    def deleteBias(self):
        nn.init.zeros_(self.bias[0])

    def initState(self, data, drop_visible=None):
        '''
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        Xdata is the the last element of the dict
        '''
        state = []
        size = data.size(0)
        for layer in range(len(self.fcLayers)-1):
            state.append(torch.zeros(size, self.fcLayers[layer], requires_grad=False))
        if drop_visible is None:
            state.append(data.float())
        else:
            state.append(drop_visible*data.float())

        return state

    def initHidden(self, data, drop_visible=None):
        h = []
        size = data.size(0)
        y = torch.zeros(size, self.fcLayers[0], requires_grad=False)
        for layer in range(1, len(self.fcLayers)-1):
            h.append(torch.zeros(size, self.fcLayers[layer], requires_grad=False))
        if drop_visible is None:
            h.append(data.float())
        else:
            h.append(drop_visible*data.float())

        return h, y


class Classifier(nn.Module):
    # one layer perceptron does not need to be trained by EP
    def __init__(self, jparams):
        super(Classifier, self).__init__()
        # construct the classifier layer
        self.classifier = torch.nn.Sequential(nn.Dropout(p=float(jparams['class_dropProb'])),
                                              nn.Linear(jparams['fcLayers'][0], jparams['n_class']),
                                              func_dict[jparams['class_activation']])

        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)

    def forward(self, x):
        return self.classifier(x)


class ConvEP(nn.Module):
    '''
    Define the network studied
    '''
    # Try to use the jit version

    def __init__(self, jparams, rho, rhop):

        super(ConvEP, self).__init__()

        self.T = jparams['T']
        self.Kmax = jparams['Kmax']
        self.dt = jparams['dt']
        self.beta = torch.tensor(jparams['beta'])
        self.clamped = jparams['clamped']
        self.errorEstimate = jparams['errorEstimate']
        self.rho = rho
        self.rhop = rhop
        self.gamma = jparams['gamma']

        if jparams['dataset'] == 'mnist':
            input_size = 28
        elif jparams['dataset'] == 'cifar10':
            input_size = 32
            # raise ValueError("The convolutional network now is only designed for mnist dataset")

        self.batchSize = jparams['batchSize']
        self.C_list = jparams['C_list']

        self.F = jparams['convF']  # filter size


        # # define padding size:
        # if jparams['padding']:
        #     pad = int((jparams['convF'] - 1)/2)
        # else:
        #     pad = 0
        #
        # self.pad = pad

        # define pooling operation
        self.pool = nn.MaxPool2d(jparams['Fpool'], stride=jparams['Fpool'], return_indices=True)
        self.unpool = nn.MaxUnpool2d(jparams['Fpool'], stride=jparams['Fpool'])

        Conv = nn.ModuleList(None)
        conv_number = len(jparams['C_list'])-1
        if conv_number < 2:
            raise ValueError("At least 2 convolutional layer should be applied")

        self.conv_number = conv_number
        self.P = jparams['Pad']
        #self.P = []
        size_conv_list = [input_size]
        size_convpool_list = [input_size]

        # define the convolutional layer
        for i in range(self.conv_number):
            Conv.extend([nn.Conv2d(jparams['C_list'][i+1], jparams['C_list'][i], jparams['convF'], padding=jparams['Pad'][i], bias=True)])
            #  in default, we use introduce the bias
            size_conv_list.append(size_convpool_list[i] - jparams['convF'] + 1 + 2*jparams['Pad'][-i-1])  # calculate the output size
            size_convpool_list.append(int(np.floor((size_convpool_list[i] - jparams['convF'] + 1 + 2*jparams['Pad'][-i-1] - jparams['Fpool'])/jparams['Fpool'] + 1)))  # the size after the pooling layer

        self.Conv = Conv

        size_conv_list = list(reversed(size_conv_list))
        self.size_conv_list = size_conv_list

        size_convpool_list = list(reversed(size_convpool_list))
        self.size_convpool_list = size_convpool_list

        # define the fully connected layer
        fcLayers = list(jparams['fcLayers'])
        fcLayers.append(jparams['C_list'][0]*size_convpool_list[0]**2)
        self.fcLayers = fcLayers
        self.fc_number = len(self.fcLayers) - 1
        self.W = nn.ModuleList(None)
        for i in range(self.fc_number):
            self.W.extend([nn.Linear(fcLayers[i + 1], fcLayers[i], bias=True)])
            # torch.nn.init.zeros_(self.W[i].bias)

        # if conv_number > 0:
        #     with torch.no_grad():
        #         for i in range(conv_number):
        #             self.Conv.extend([nn.Conv2d(args.convLayers[(conv_number - i - 1) * 5],
        #                                         args.convLayers[(conv_number - i - 1) * 5 + 1],
        #                                         args.convLayers[(conv_number - i - 1) * 5 + 2],
        #                                         stride=args.convLayers[(conv_number - i - 1) * 5 + 3],
        #                                         padding=args.convLayers[(conv_number - i - 1) * 5 + 4], bias=True)])
        #             self.P.append(args.convLayers[(conv_number - i - 1) * 5 + 4])
        #
        # self.unConv = nn.ModuleList(None)
        # if conv_number > 1:
        #     with torch.no_grad():
        #         for i in range(conv_number - 1):
        #             self.unConv.extend([nn.Conv2d(args.convLayers[(conv_number - i - 1) * 5 + 1],
        #                                           args.convLayers[(conv_number - i - 1) * 5],
        #                                           args.convLayers[(conv_number - i - 1) * 5 + 2],
        #                                           stride=args.convLayers[(conv_number - i - 1) * 5 + 3],
        #                                           padding=args.convLayers[(conv_number - i - 1) * 5 + 2] - 1 -
        #                                                   args.convLayers[(conv_number - i - 1) * 5 + 4], bias=False)])

        # # We test 2 different Xavier Initiation
        # inf = []
        # for i in range(len(args.layersList)-1):
        #     inf.append(np.sqrt(1/args.layersList[i+1]))
        #     torch.nn.init.uniform_(self.W[i].weight, a=-inf[i], b=inf[i])
        #     torch.nn.init.normal_(self.W[i].bias, 0, 0.05)

        # put model on GPU is available and asked
        if jparams['device'] >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(jparams['device']))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)

    def stepper_p_conv(self, s, data, P_ind, p_distribut=None, target=None, beta=0, return_derivatives=False):
        '''
        stepper function for prototypical convolutional model of EP
        '''
        data = data.float()
        dsdt = []

        # fully connected layer (classifier part)
        dsdt.append(-s[0]+self.rho(self.W[0](s[1].view(s[1].size(0), -1))))  # flatten the s[1]? does it necessary?

        if beta != 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, len(self.fcLayers)-1):
            dsdt.append(-s[i] + self.rho(self.W[i](s[i+1].view(s[i+1].size(0), -1)) + torch.mm(s[i-1], self.W[i-1].weight)))
            # at the same time we flatten the layer before

        # Convolutional part
        # last conv layer
        s_pool, ind = self.pool(self.Conv[0](s[self.fc_number+1]))
        P_ind[0] = ind  # len(P_ind) = conv_number
        dsdt.append(-s[self.fc_number] + self.rho(s_pool + torch.mm(s[self.fc_number-1], self.W[-1].weight).view(s[self.fc_number].size()))) # unflatten the result
        del s_pool, ind

        # middle conv layers
        for i in range(1, self.conv_number-1):
            s_pool, ind = self.pool(self.Conv[i](s[self.fc_number+1+i]))
            P_ind[i] = ind

            if P_ind[i-1] is not None:
                output_size = [s[self.fc_number+i-1].size(0), s[self.fc_number+i-1].size(0), self.size_conv_list[i-1],
                               self.size_conv_list[i-1]]
                s_unpool = F.conv_transpose2d(self.unpool(s[self.fc_number+i-1], P_ind[i-1], output_size=output_size),
                                              weight=self.Conv[i-1].weight, padding=self.P[i-1])
                dsdt.append(-s[self.fc_number+i] + self.rho(s_pool + s_unpool))
                del s_pool, s_unpool, ind, output_size

        # first conv layer
        s_pool, ind = self.pool(self.Conv[-1](data))
        P_ind[-1] = ind
        if P_ind[-2] is not None:
            output_size = [s[-2].size(0), s[-2].size(1), self.size_conv_list[-3], self.size_conv_list[-3]]
            s_unpool = F.conv_transpose2d(self.unpool(s[-2], P_ind[-2], output_size=output_size),
                                          weight=self.Conv[-2].weight, padding=self.P[-2])
            dsdt.append(-s[-1] + self.rho(s_pool + s_unpool))
            del s_pool, s_unpool, ind, output_size

        for i in range(len(s)):
            s[i] = s[i] + self.dt*dsdt[i]
            if p_distribut is not None:
                s[i] = p_distribut[i]*s[i]

        if return_derivatives:
            return s, P_ind, dsdt
        else:
            del dsdt
            return s, P_ind

    def stepper_generate(self, s: List[torch.Tensor], target: torch.Tensor):
        # TODO set the generate step for CNN

        dsdt = []
        # fix the output
        s[0] = target.clone()
        # dsdt.append(0.5 * (target - s[0]))
        for layer in range(1, len(s) - 1):
            dsdt.append(-s[layer] + self.rhop(s[layer]) * (torch.mm(self.rho(s[layer + 1]), self.W[layer]) +
                                                           self.bias[layer] + torch.mm(self.rho(s[layer - 1]),
                                                                                       self.W[layer - 1].T)))

        # for the input layer
        dsdt.append(-s[-1] + (self.rhop(s[-1])) * torch.mm(self.rho(s[-2]), self.W[-1].T))  # no biases

        for (layer, dsdt_item) in enumerate(dsdt):
            s[layer + 1] = s[layer + 1] + self.dt * dsdt_item
            s[layer + 1] = s[layer + 1].clamp(0, 1)

        return s

    def generate_image(self, clamp_input, target):
        # TODO finish the generation cycle
        s, P_ind = self.initHidden(clamp_input)
        # transfer to cuda
        if self.cuda:
            s = [item.to(self.device) for item in s]
        with torch.no_grad():
            for t in range(self.T):
                s = self.stepper_generate(s, target)
        return s

    def forward(self, s, data, P_ind, p_distribut=None, beta=0, target=None, tracking=False):
        # TODO why rename the self.variable at the beginning?
        T, Kmax = self.T, self.Kmax
        n_track = 9
        m, n, p, q = [[] for k in range(n_track)], [[] for k in range(n_track)], \
                     [[] for k in range(n_track)], [[] for k in range(n_track)]

        # TODO verify the tracking part
        with torch.no_grad():
            if beta == 0:
                # first phase
                for T in range(T):
                    s, P_ind = self.stepper_p_conv(s, data, P_ind, p_distribut)
                    if tracking:
                        for k in range(n_track):
                            m[k].append(s[0][k][2*k].item())
                            n[k].append(s[1][k][2*k].item())
                            p[k].append(s[2][k][2][k][k])  # out_channel=2
                            q[k].append(s[3][k][3][k][k])  # out_channel=3
            else:
                # nudging phase
                for K in range(Kmax):
                    s, P_ind = self.stepper_p_conv(s, data, P_ind, p_distribut, target=target, beta=beta)
                    if tracking:
                        for k in range(n_track):
                            m[k].append(s[0][k][2 * k].item())
                            n[k].append(s[1][k][2 * k].item())
                            p[k].append(s[2][k][2][k][k])  # out_channel=2
                            q[k].append(s[3][k][3][k][k])  # out_channel=3

        if tracking:
            return s, P_ind, m, n, p, q
        else:
            return s, P_ind

        # TODO consider C-EP (or the other methods) in the future?

    def computeConvGradientEP(self, data, s, seq, P_ind, Peq_ind):
        batch_size = s[0].size(0)
        coef = 1 / (self.beta * batch_size)
        if self.errorEstimate == 'symmetric':
            coef = coef*0.5

        gradW_fc, gradBias_fc = [], []
        gradW_conv, gradBias_conv = [], []

        with torch.no_grad():
            # classifier
            for i in range(self.fc_number-1):
                gradW_fc.append(coef*(torch.mm(torch.transpose(s[i], 0, 1), s[i + 1]) - torch.mm(torch.transpose(seq[i],0, 1),
                                                                                   seq[i + 1])))
                gradBias_fc.append(coef*(s[i]-seq[i]).sum(0))

            # update the last layer of fc
            gradW_fc.append(coef * (torch.mm(torch.transpose(s[self.fc_number-1], 0, 1), s[self.fc_number].view(s[self.fc_number].size(0), -1)) -
                                        torch.mm(torch.transpose(seq[self.fc_number-1], 0, 1), seq[self.fc_number].view(s[self.fc_number].size(0), -1))))

            gradBias_fc.append(coef * (s[self.fc_number-1] - seq[self.fc_number-1]).sum(0))

            # convolutional layers
            for i in range(self.conv_number-1):
                output_size = [s[self.fc_number+i].size(0), s[self.fc_number+i].size(1), self.size_conv_list[i], self.size_conv_list[i]]

                gradW_conv.append(coef*(F.conv2d(s[self.fc_number + i + 1].permute(1,0,2,3),
                                                 self.unpool(s[self.fc_number + i], P_ind[i], output_size=output_size).permute(1,0,2,3), padding=self.P[i])
                                        - F.conv2d(seq[self.fc_number + i + 1].permute(1,0,2,3),
                                                   self.unpool(seq[self.fc_number+i], Peq_ind[i], output_size=output_size).permute(1,0,2,3), padding=self.P[i])).permute(1, 0, 2, 3))
                gradBias_conv.append(coef*(self.unpool(s[self.fc_number+i], P_ind[i], output_size=output_size) -
                                           self.unpool(seq[self.fc_number+i], Peq_ind[i], output_size=output_size)).permute(1,0,2,3).contiguous().view(s[self.fc_number+i].size(1), -1).sum(1))

            # last layer
            output_size = [s[-1].size(0), s[-1].size(1), self.size_conv_list[-2], self.size_conv_list[-2]]
            gradW_conv.append(coef*(F.conv2d(data.permute(1,0,2,3), self.unpool(s[-1], P_ind[-1], output_size=output_size).permute(1,0,2,3), padding=self.P[-1])-
                                    F.conv2d(data.permute(1,0,2,3), self.unpool(seq[-1], Peq_ind[-1], output_size=output_size).permute(1,0,2,3), padding=self.P[-1])).permute(1,0,2,3))
            gradBias_conv.append(coef*(self.unpool(s[-1], P_ind[-1], output_size=output_size)-self.unpool(seq[-1], Peq_ind[-1], output_size=output_size)).permute(1,0,2,3).contiguous().view(s[-1].size(1),-1).sum(1))

        for (i, param) in enumerate(self.W):
            param.weight.grad = -gradW_fc[i]
            param.bias.grad = -gradBias_fc[i]
        for (i, param) in enumerate(self.Conv):
            param.weight.grad = -gradW_conv[i]
            param.bias.grad = -gradBias_conv[i]



    def initHidden(self, batch_size):
        s = []
        P_ind = []
        for i in range(self.fc_number):
            s.append(torch.zeros(batch_size, self.fcLayers[i], requires_grad=False))  # why we requires grad ? for comparison with BPTT?
            # inds.append(None)
        for i in range(self.conv_number):
            s.append(torch.zeros(batch_size, self.C_list[i], self.size_convpool_list[i], self.size_convpool_list[i],
                                 requires_grad=False))
            P_ind.append(None)

        return s, P_ind





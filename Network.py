# File defining the network and the oscillators composing the network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit

from main import rho, rhop
from typing import List, Optional, Tuple



'''
Try to use the same function name for the MLP class and Conv class
but the problem is that, there are different variables for Conv class than the MLP class,
so whether we will merge s and h together?
'''


class MlpEP(jit.ScriptModule):

    def __init__(self, args):

        super(MlpEP, self).__init__()

        self.T = args.T
        self.Kmax = args.Kmax
        self.dt = args.dt
        self.beta = torch.tensor(args.beta)
        self.clamped = args.clamped
        self.lr = args.lr
        self.coeffDecay = args.coeffDecay
        self.epochDecay = args.epochDecay
        self.batchSize = args.batchSize
        self.gamma = args.gamma
        self.fcLayers = args.fcLayers
        self.errorEstimate = args.errorEstimate

        # define the device
        if args.device >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(args.device))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        # The following parameters are for
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.m_dw, self.v_dw = [], []
        self.m_db, self.v_db = [], []
        with torch.no_grad():
            for i in range(len(args.fcLayers)-1):
                self.m_dw.append(torch.zeros(args.fcLayers[i+1], args.fcLayers[i], device=device))
                self.v_dw.append(torch.zeros(args.fcLayers[i+1], args.fcLayers[i], device=device))
                self.m_db.append(torch.zeros(args.fcLayers[i], device=device))
                self.v_db.append(torch.zeros(args.fcLayers[i], device=device))
        self.epsillon = 1e-8

        # We define the list to save the weights
        W:List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(len(args.fcLayers)-1):
                w = torch.empty(args.fcLayers[i+1], args.fcLayers[i], device=device)
                bound = 1/ np.sqrt(args.fcLayers[i+1])
                #nn.init.xavier_uniform_(w)
                nn.init.uniform_(w, a=-bound, b=bound)
                W.append(w)
        self.W = W

        # We define the list to save the bias
        bias:List[torch.Tensor] = []
        with torch.no_grad():
            for i in range(len(args.fcLayers)-1):
                b = torch.empty(args.fcLayers[i], device=device)
                bound = 1/np.sqrt(args.fcLayers[1])
                #nn.init.uniform_(b, a=-bound, b=bound)
                nn.init.zeros_(b)
                bias.append(b)
        self.bias = bias

        self = self.to(device)

    #@jit.script_method
    def mydropout(self, s:List[torch.Tensor], p:List[float])->List[torch.FloatTensor]:
        # if p < 0 or p > 1:
        #     raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        p_distribut = []
        # bernoulli = torch.distributions.bernoulli.Bernoulli(total_count=1, probs=1-p)

        for layer in range(len(s)-1):
            binomial = torch.distributions.binomial.Binomial(probs=torch.tensor(1-p[layer]))
            p_distribut.append(binomial.sample(s[layer].size()))
        return p_distribut

    @jit.script_method
    def stepper_hidden(self, h:List[torch.Tensor], target:Optional[torch.Tensor]=None,
                          beta:Optional[float]=None)->Tuple[List[torch.Tensor], torch.Tensor]:

        y = F.softmax(torch.mm(rho(h[0]), self.W[0]) + self.bias[0], dim=1)
        if len(h) > 1:
            dhdt=[]
            dhdt.append(-h[0] + (rhop(h[0]) * (torch.mm(rho(h[1]), self.W[1]) + self.bias[1])))
            if target is not None and beta is not None:
                dhdt[0] = dhdt[0] + beta * torch.mm((target-y), self.W[0].T)

            for layer in range(1, len(h)-1):
                dhdt.append(-h[layer] + rhop(h[layer]) * (
                            torch.mm(rho(h[layer + 1]), self.W[layer+1]) + self.bias[layer+1] + torch.mm(rho(h[layer - 1]),
                                                                                                     self.W[layer].T)))

            for (layer, dhdt_item) in enumerate(dhdt):
                h[layer] = h[layer]+self.dt*dhdt_item
                if self.clamped:
                    h[layer] = h[layer].clamp(0, 1)

        return h, y

    @jit.script_method
    def forward_softmax(self, h:List[torch.Tensor], beta:Optional[float]=None, target:Optional[torch.Tensor]=None,
                ) -> Tuple[List[torch.Tensor], torch.Tensor]:

        T, Kmax = self.T, self.Kmax

        y = F.softmax(torch.mm(rho(h[0]), self.W[0]) + self.bias[0], dim=1)

        with torch.no_grad():
            if beta is None and target is None:
                if len(h) > 1:
                    for t in range(T):
                        h, y = self.stepper_hidden(h, target=target, beta=beta)
            else:
                for t in range(Kmax):
                    h, y = self.stepper_hidden(h, target=target, beta=beta)
        return h, y

    @jit.script_method
    def stepper_c_ep(self, s:List[torch.Tensor], target:Optional[torch.Tensor]=None,
                     beta:Optional[float]=None):
        '''
        stepper function for energy-based dynamics of EP
        '''
        dsdt = []

        dsdt.append(-s[0] + (rhop(s[0])*(torch.mm(rho(s[1]), self.W[0]) + self.bias[0])))

        if target is not None and beta is not None:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for layer in range(1, len(s)-1):  # start at the first hidden layer and then to the before last hidden layer
            dsdt.append(-s[layer] + rhop(s[layer])*(torch.mm(rho(s[layer+1]), self.W[layer])+self.bias[layer]+torch.mm(rho(s[layer-1]), self.W[layer-1].T)))

        for (layer, dsdt_item) in enumerate(dsdt):
            s[layer] = s[layer] + self.dt*dsdt_item
            # s[0] = s[0].clamp(0, 1)
            if self.clamped:
                s[layer] = s[layer].clamp(0, 1)

        return s

    @jit.script_method
    def dropout_stepper(self, s:List[torch.Tensor], p_distribut:Optional[List[torch.Tensor]], target:Optional[torch.Tensor]=None, beta:Optional[float]=None):
        '''
        stepper function for the network
        '''

        dsdt = []

        dsdt.append(-s[0] + (rhop(s[0]) * (torch.mm(rho(s[1]), self.W[0]) + self.bias[0])))

        if target is not None and beta is not None:
            dsdt[0] = dsdt[0] + beta * (target - s[0])

        for layer in range(1, len(s) - 1):  # start at the first hidden layer and then to the before last hidden layer
            dsdt.append(-s[layer] + rhop(s[layer]) * (
                        torch.mm(rho(s[layer + 1]), self.W[layer]) + self.bias[layer] + torch.mm(rho(s[layer - 1]),
                                                                                                 self.W[layer - 1].T)))

        for (layer, dsdt_item) in enumerate(dsdt):
            if p_distribut is not None:
                s[layer] = p_distribut[layer]*(s[layer] + self.dt * dsdt_item)
            else:
                s[layer] = s[layer] + self.dt * dsdt_item

            # s[0] = s[0].clamp(0, 1)
            if self.clamped:
                s[layer] = s[layer].clamp(0, 1)

        return s

    @jit.script_method
    def forward(self, s:List[torch.Tensor], p_distribut:Optional[List[torch.Tensor]]=None, beta:Optional[float]=None, target:Optional[torch.Tensor]=None,
                ) -> List[torch.Tensor]:

        T, Kmax = self.T, self.Kmax

        with torch.no_grad():
            # continuous time EP
            if p_distribut is None:
                if beta is None and target is None:
                    # free phase
                    # TODO this if selection can be improved by giving the T/Kmax outside
                    for t in range(T):
                        s = self.stepper_c_ep(s, target=target, beta=beta)

                else:
                    # nudged phase
                    for t in range(Kmax):
                        s = self.stepper_c_ep(s, target=target, beta=beta)
            else:
                if beta is None and target is None:
                    # free phase
                    for t in range(T):
                        s = self.dropout_stepper(s, p_distribut, target=target, beta=beta)

                else:
                    # nudged phase
                    for t in range(Kmax):
                        s = self.dropout_stepper(s, p_distribut, target=target, beta=beta)

        return s

    @jit.script_method
    def computeGradientsEP(self, s:List[torch.Tensor], seq:List[torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
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

        alpha = 0.1
        gradW, gradBias = [], []

        with torch.no_grad():
            for layer in range(len(s)-1):
                gradW.append(coef*(torch.mm(torch.transpose(rho(s[layer+1]), 0, 1), rho(s[layer]))
                                   -torch.mm(torch.transpose(rho(seq[layer+1]), 0, 1), rho(seq[layer]))))
                gradBias.append(coef*(rho(s[layer])-rho(seq[layer])).sum(0))
            # for layer in range(len(s)-1):
            #     gradW.append(coef*(torch.mm(torch.transpose(rho(s[layer+1]), 0, 1), rho(s[layer]))
            #                        -torch.mm(torch.transpose(rho(seq[layer+1]), 0, 1), rho(seq[layer]))) -
            #                  alpha*self.W[layer])
            #     gradBias.append(coef*(rho(s[layer])-rho(seq[layer])).sum(0)-alpha*self.bias[layer])
        return gradW, gradBias

    @jit.script_method
    def computeGradientEP_softmax(self, h:List[torch.Tensor], heq:List[torch.Tensor], y:torch.Tensor, target:torch.Tensor,
                             ybeta:Optional[torch.Tensor]=None):
        # define the coefficient for the hidden neurons
        batch_size = h[0].size(0)
        coef = 1 / (self.beta * batch_size)
        if self.errorEstimate == 'symmetric':
            coef = coef * 0.5
        gradW, gradBias = [], []

        with torch.no_grad():
            if ybeta is None:
                gradW.append(-torch.mm(torch.transpose(rho(h[0]), 0, 1), (y-target)))
                gradBias.append(-(y-target).sum(0))
            else:
                gradW.append(-0.5*(torch.mm(torch.transpose(rho(h[0]), 0, 1), (y-target)) +
                                          torch.mm(torch.transpose(rho(heq[0]), 0, 1), (ybeta-target))))
                gradBias.append(-0.5*(y+ybeta-2*target).sum(0))
            for layer in range(len(h)-1):
                gradW.append(coef * (torch.mm(torch.transpose(rho(h[layer + 1]), 0, 1), rho(h[layer]))
                                     - torch.mm(torch.transpose(rho(heq[layer + 1]), 0, 1), rho(heq[layer]))))
                gradBias.append(coef * (rho(h[layer]) - rho(heq[layer])).sum(0))
        return gradW, gradBias

    @jit.script_method
    def updateWeight_softmax(self, h:List[torch.Tensor], heq:List[torch.Tensor], y:torch.Tensor, target:torch.Tensor,
                             ybeta:Optional[torch.Tensor]=None, epoch=1):

        '''update the weights and biases of network with a softmax output'''

        gradW, gradBias = self.computeGradientEP_softmax(h, heq, y, target, ybeta=ybeta)
        with torch.no_grad():
            # update the hidden layers
            for layer in range(len(self.W)):
                lrDecay = self.lr[layer] * torch.pow(self.coeffDecay, int(epoch / self.epochDecay))
                self.W[layer] += lrDecay * gradW[layer]
                self.bias[layer] += lrDecay * gradBias[layer]

    @jit.script_method
    def Adam_updateWeight_softmax(self, h:List[torch.Tensor], heq:List[torch.Tensor], y:torch.Tensor, target:torch.Tensor,
                             ybeta:Optional[torch.Tensor]=None, epoch=1):
        gradW, gradBias = self.computeGradientEP_softmax(h, heq, y, target, ybeta=ybeta)

        m_dw_new, m_db_new, v_dw_new, v_db_new = [], [], [], []
        # update weights and bias
        with torch.no_grad():
            # lrDecay = 0.01*np.power(0.97, epoch)
            # lrDecay = 0.01*np.power(0.5, int(epoch/10))

            for layer in range(len(self.W)):
                # calculate the iteration momentum and velocity
                m_dw_new.append(self.beta1 * self.m_dw[layer] + (1 - self.beta1) * gradW[layer])
                m_db_new.append(self.beta1 * self.m_db[layer] + (1 - self.beta1) * gradBias[layer])
                # self.m_dw[layer] = self.beta1 * self.m_dw[layer] + (1 - self.beta1) * gradW[layer]  # momentum for weights
                # self.m_db[layer] = self.beta1 * self.m_db[layer] + (1 - self.beta1) * gradBias[layer]  # momentum for bias

                v_dw_new.append(self.beta2 * self.v_dw[layer] + (1 - self.beta2) * (gradW[layer] ** 2))
                v_db_new.append(self.beta2 * self.v_db[layer] + (1 - self.beta2) * (gradBias[layer] ** 2))

                # self.v_dw[layer] = self.beta2 * self.v_dw[layer] + (1 - self.beta2) * (gradW[layer]**2)  # velocity for weights
                # self.v_db[layer] = self.beta2 * self.v_db[layer] + (1 - self.beta2) * gradBias[layer]  # velocity for bias

                # bias correction
                m_dw_corr = m_dw_new[layer] / (1 - self.beta1 ** epoch)
                m_db_corr = m_db_new[layer] / (1 - self.beta1 ** epoch)
                v_dw_corr = v_dw_new[layer] / (1 - self.beta2 ** epoch)
                v_db_corr = v_db_new[layer] / (1 - self.beta2 ** epoch)

                # update the weight
                # TODO solve the problem of torch.pow or np.power
                lrDecay = self.lr[layer] * torch.pow(torch.tensor(self.coeffDecay),
                                                     torch.tensor(int(epoch / self.epochDecay)))
                # print('the decayed lr is:', lrDecay)
                # print('alpha is:', alpha[layer])
                self.W[layer] += lrDecay * (m_dw_corr / (torch.sqrt(v_dw_corr) + self.epsillon))
                self.bias[layer] += lrDecay * (m_db_corr / (torch.sqrt(v_db_corr) + self.epsillon))
        self.m_dw = m_dw_new
        self.m_db = m_db_new
        self.v_dw = v_dw_new
        self.v_db = v_db_new

    @jit.script_method
    def updateWeight(self, s:List[torch.Tensor], seq:List[torch.Tensor], epoch=1):
        '''
        Update weights and bias according to EQ algo
        '''

        gradW, gradBias = self.computeGradientsEP(s, seq)

        with torch.no_grad():
            #lrDecay = 0.01*np.power(0.97, epoch)
            #lrDecay = 0.01*np.power(0.5, int(epoch/10))
            for layer in range(len(self.W)):
                #lrDecay = self.lr[layer]
                # TODO solve the problem of torch.pow or np.power
                lrDecay = self.lr[layer]*torch.pow(self.coeffDecay, int(epoch/self.epochDecay))
                #print('the decayed lr is:', lrDecay)
                # print('alpha is:', alpha[layer])

                self.W[layer] += lrDecay*gradW[layer]
                self.bias[layer] += lrDecay*gradBias[layer]


    @jit.script_method
    def Adam_updateWeight(self, s:List[torch.Tensor], seq:List[torch.Tensor], epoch=1):
        '''
        Update weights using the Adam optimizer
        '''
        #
        # m_dw_before: List[torch.Tensor], m_db_before: List[torch.Tensor],
        # v_dw_before: List[torch.Tensor], v_db_before: List[torch.Tensor],
        # calculate the gradients
        gradW, gradBias = self.computeGradientsEP(s, seq)

        m_dw_new, m_db_new, v_dw_new, v_db_new = [], [], [], []
        # update weights and bias
        with torch.no_grad():
            # lrDecay = 0.01*np.power(0.97, epoch)
            # lrDecay = 0.01*np.power(0.5, int(epoch/10))

            for layer in range(len(self.W)):
                # calculate the iteration momentum and velocity
                m_dw_new.append(self.beta1 * self.m_dw[layer] + (1 - self.beta1) * gradW[layer])
                m_db_new.append(self.beta1 * self.m_db[layer] + (1 - self.beta1) * gradBias[layer])
                #self.m_dw[layer] = self.beta1 * self.m_dw[layer] + (1 - self.beta1) * gradW[layer]  # momentum for weights
                #self.m_db[layer] = self.beta1 * self.m_db[layer] + (1 - self.beta1) * gradBias[layer]  # momentum for bias

                v_dw_new.append(self.beta2 * self.v_dw[layer] + (1 - self.beta2) * (gradW[layer]**2))
                v_db_new.append(self.beta2 * self.v_db[layer] + (1 - self.beta2) * (gradBias[layer]**2))

                #self.v_dw[layer] = self.beta2 * self.v_dw[layer] + (1 - self.beta2) * (gradW[layer]**2)  # velocity for weights
                #self.v_db[layer] = self.beta2 * self.v_db[layer] + (1 - self.beta2) * gradBias[layer]  # velocity for bias

                # bias correction
                m_dw_corr = m_dw_new[layer] / (1 - self.beta1 ** epoch)
                m_db_corr = m_db_new[layer] / (1 - self.beta1 ** epoch)
                v_dw_corr = v_dw_new[layer] / (1 - self.beta2 ** epoch)
                v_db_corr = v_db_new[layer] / (1 - self.beta2 ** epoch)

                # update the weight
                # TODO solve the problem of torch.pow or np.power
                lrDecay = self.lr[layer] * torch.pow(torch.tensor(self.coeffDecay), torch.tensor(int(epoch / self.epochDecay)))
                # print('the decayed lr is:', lrDecay)
                # print('alpha is:', alpha[layer])
                self.W[layer] += lrDecay*(m_dw_corr/(torch.sqrt(v_dw_corr) + self.epsillon))
                self.bias[layer] += lrDecay*(m_db_corr/(torch.sqrt(v_db_corr) + self.epsillon))
        self.m_dw = m_dw_new
        self.m_db = m_db_new
        self.v_dw = v_dw_new
        self.v_db = v_db_new

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


        # # we devide the weight by the maxmium possible value
        # max_value = torch.max(self.W[0])
        # if max_value > 0:
        #  self.W[0] = self.W[0]/max_value

    @jit.script_method
    def deleteBias(self):
        nn.init.zeros_(self.bias[0])

    # def softmax_target(self, output):

    def unsupervised_target(self, output, N, Xth=None):

        # define unsupervised target
        unsupervised_targets = torch.zeros(output.size(), device=self.device)

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
        unsupervised_targets.scatter_(1, N_maxindex, torch.ones(output.size(), device=self.device))
        # print('the unsupervised vector is:', unsupervised_targets)

        return unsupervised_targets, N_maxindex

    def initState(self, fcLayers, data):
        '''
        Init the state of the network
        State if a dict, each layer is state["S_layer"]
        Xdata is the the last element of the dict
        '''
        state = []
        size = data.size(0)
        for layer in range(len(fcLayers)-1):
            state.append(torch.zeros(size, fcLayers[layer], requires_grad=False))

        state.append(data.float())

        return state

    def initHidden(self, fcLayers, data):
        h = []
        size = data.size(0)
        # y = torch.zeors(size, fcLayers[0], requires_grad=False)
        for layer in range(1, len(fcLayers)-1):
            h.append(torch.zeros(size, fcLayers[layer], requires_grad=False))

        h.append(data.float())

        return h





# class Dropout(nn.Module):
#     def __init__(self, p: float = 0.5):
#         super(Dropout, self).__init__()
#         if p < 0 or p > 1:
#             raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
#         self.p = p
#
#     def forward(self, s:List[torch.Tensor])->List[torch.Tensor]:
#         # return a binary vector for the training process
#         #binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
#         p_distribut = []
#         binomial = torch.distributions.binomial.Binomial(probs=torch.tensor(1-self.p))
#         #     #bernoulli = torch.distributions.bernoulli.Bernoulli(prob=1-p)
#         for layer in range(len(s)-1):
#             p_distribut.append(binomial.sample(s[layer].size()) * (1.0 / (1 - self.p)))
#
#         p_distribut.append(binomial.sample(s[layer].size()) * (1.0 / (1 - self.p)))
#
#         return p_distribut


class ConvEP(nn.Module):
    '''
    Define the network studied
    '''

    def __init__(self, args):

        super(ConvEP, self).__init__()

        self.T = args.T
        self.Kmax = args.Kmax
        self.dt = args.dt
        self.beta = torch.tensor(args.beta)
        self.clamped = args.clamped
        self.lr = args.lr

        if args.dataset == 'mnist':
            input_size = 28
        else:
            raise ValueError("The convolutional network now is only designed for mnist dataset")

        self.batchSize = args.batchSize
        self.C_list = args.C_list

        self.F = args.convF  # filter size

        # define padding size:
        if args.padding:
            pad = int((args.F - 1)/2)
        else:
            pad = 0

        self.pad = pad

        # define pooling operation
        self.pool = nn.MaxPool2d(args.Fpool, stride=args.Fpool, return_indices=True)
        self.unpool = nn.MaxUnpool2d(args.Fpool, stride=args.Fpool)

        Conv = nn.ModuleList(None)
        conv_number = len(args.C_list)-1
        if conv_number < 2:
            raise ValueError("At least 2 convolutional layer should be applied")

        self.conv_number = conv_number
        #self.P = []
        size_conv_list = [input_size]
        size_convpool_list = [input_size]

        # define the convolutional layer
        with torch.no_grad():
            for i in range(self.conv_number):
                Conv.extend([nn.Conv2d(args.C_list[i+1], args.C_list[i], args.convF, bias=True)])
                #  in default, we use introduce the bias
                size_conv_list.append(size_conv_list[i] - args.convF + 1 + 2*pad)  # calculate the output size
                size_convpool_list.append(int(np.floor((size_convpool_list[i] - args.convF + 1 + 2*pad - args.Fpool)/args.Fpool + 1)))  # the size after the pooling layer

        self.Conv = Conv

        size_conv_list = list(reversed(size_conv_list))
        self.size_conv_list = size_conv_list

        size_convpool_list = list(reversed(size_convpool_list))
        self.size_convpool_list = size_convpool_list

        # define the fully connected layer
        fcLayers = list(args.fcLayers)
        fcLayers.append(args.C_tab[0]*size_convpool_list[0]**2)
        self.fcLayers = fcLayers
        self.fc_number = len(self.fcLayers) - 1

        self.W = nn.ModuleList(None)
        with torch.no_grad():
            for i in range(self.fc_number):
                self.W.extend([nn.Linear(args.fcLayers[i + 1], args.fcLayers[i], bias=True)])
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
        if args.device >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(args.device))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device
        self = self.to(device)

    def stepper_p_conv(self, s, data, P_ind, target=None, beta=0, return_derivatives = False):
        '''
        stepper function for prototypical convolutional model of EP
        '''

        dsdt = []

        # fully connected layer (classifier part)
        dsdt.append(-s[0]+rho(self.W[0](s[1].view(s[1].size(0), -1))))  # flatten the s[1]? does it necessary?

        if beta != 0:
            dsdt[0] = dsdt[0] + beta*(target-s[0])

        for i in range(1, len(self.fcLayers)-1):
            dsdt.append(-s[i] + rho(self.W[i](s[i+1]).view(s[i+1].size(0), -1), torch.mm(s[i-1], self.W[i-1].weight)))
            # at the same time we flatten the layer before

        # Convolutional part
        # last conv layer
        s_pool, ind = self.pool(self.Conv[0](s[self.fc_number+1]))
        P_ind[0] = ind # len(P_ind) = conv_number
        dsdt.append(-s[self.fc_number] + rho(s_pool + torch.mm(s[self.fc_number-1], self.W[-1].weight).view(s[self.fc_number].size()))) # unflatten the result
        del s_pool, ind

        # middle conv layers
        for i in range(1, self.conv_number-1):
            s_pool, ind = self.pool(self.Conv[i](s[self.fc_number+1+i]))
            P_ind[i] = ind

            if P_ind[i-1] is not None:
                output_size = [s[self.fc_number+i-1].size(0), s[self.fc_number+i-1].size(0), self.size_conv_list[i-1],
                               self.size_conv_list[i-1]]
                s_unpool = F.conv_transpose2d(self.unpool(s[self.fc_number+i-1], P_ind[i-1], output_size=output_size),
                                              weight=self.Conv[i-1].weight, padding=self.pad)
                dsdt.append(-s[self.fc_number+i] + rho(s_pool + s_unpool))
                del s_pool, s_unpool, ind, output_size

        # first conv layer
        s_pool, ind = self.pool(self.Conv[-1](data))
        P_ind[-1] = ind
        if P_ind[-2] is not None:
            output_size = [s[-2].size(0), s[-2].size(1), self.size_conv_list[-3], self.size_conv_list[-3]]
            s_unpool = F.conv_transpose2d(self.unpool(s[-2], P_ind[-2], output_size=output_size),
                                          weight=self.Conv[-2].weight, padding=self.pad)
            dsdt.append(-s[1] + rho(s_pool + s_unpool))
            del s_pool, s_unpool, ind, output_size

        for i in range(len(s)):
            s[i] = s[i] + self.dt*dsdt[i]

        if return_derivatives:
            return s, P_ind, dsdt
        else:
            del dsdt
            return s, P_ind

        # snew = []
        # snew.append(rho(self.W[0](s[1])))
        #
        # if beta != 0:
        #     snew[0] = snew[0] + beta*(target-s[0])  # prototypical model (without the damping of s)
        #
        # if len(s) > 2:
        #     for fc in range(1, len(s)-2):
        #         snew.append(rho(self.W[fc](s[fc+1]) + torch.mm(s[fc-1], self.W[fc-1].weight)))
        #
        # snew.append(rho(self.W[-1](h[0].flatten(1,-1)) + torch.mm(s[-2], self.W[-2].weight)))
        #
        # hnew = []
        #
        # h0, ind = self.Pool(self.Conv[0](h[1]))
        # P_ind[0] = ind
        # hnew.append(rho(h0 + (torch.mm(s[-1], self.W[-1].weight)).unflatten(1, h[0].size()[1:])))
        # del ind, h0
        #
        # # save the input images at h[-1], so we do not update h[-1]
        # if len(h) > 2:
        #     for cv in range(1, len(s)-2):
        #         hcv, ind = self.Pool(self.Conv[cv](h[cv+1]))
        #         P_ind[cv] = ind
        #         # update the unConv weight data
        #         self.unConv[cv-1].weight.data = (self.Conv[cv-1].weight.flip([2, 3])).transpose(0, 1)
        #
        #         hnew.append(hcv + self.unConv[cv-1](self.unPool(h[cv-1], P_ind[cv-1])))
        #         del hcv, ind
        #
        # # update
        # s, h = snew, hnew

        # return s, h, P_ind

    def forward(self, s, data, P_ind, beta=0, target=None, tracking=False):
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
                    s, P_ind = self.stepper_p_conv(s, data, P_ind)
                    if tracking:
                        for k in range(n_track):
                            m[k].append(s[0][k][2*k].item())
                            n[k].append(s[1][k][2*k].item())
                            p[k].append(s[2][k][2][k][k])  # out_channel=2
                            q[k].append(s[3][k][3][k][k])  # out_channel=3
            else:
                # nudging phase
                for K in range(Kmax):
                    s, P_ind = self.stepper_p_conv(s, data, P_ind, target=target, beta=beta)
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

            # if len(s) > 2:
            #     for fc in range(len(s)-2):
            #         gradW_fc.append(coef*(torch.mm(torch.transpose(s[fc], 0, 1), s[fc+1]) -
            #                              torch.mm(torch.transpose(seq[fc], 0, 1), seq[fc+1])))
            #         gradBias_fc.append(coef*(s[fc]-seq[fc]).sum(0))

            # convolutional layers
            for i in range(self.conv_number-1):
                output_size = [s[self.fc_number+i].size(0), s[self.fc_number+i].size(1), self.size_conv_list[i], self.size_conv_list[i]]

                gradW_conv.append(coef*(F.conv2d(s[self.fc_number + i + 1].permute(1,0,2,3),
                                                 self.unpool(s[self.fc_number + i], P_ind[self.fc_number + i], output_size=output_size).permute(1,0,2,3), padding=self.pad)
                                        - F.conv2d(seq[self.fc_number + i + 1].permute(1,0,2,3),
                                                   self.unpool(seq[self.fc_number+i], Peq_ind[self.fc_number+i], output_size=output_size).permute(1,0,2,3), padding=self.pad)).permute(1, 0, 2, 3))
                gradBias_conv.append(coef*(self.unpool(s[self.fc_number+i], P_ind[self.fc_number+i], output_size=output_size) -
                                           self.unpool(seq[self.fc_number+i], Peq_ind[self.fc_number+i], output_size=output_size)).permute(1,0,2,3).contiguous().view(s[self.fc_number+i].size(1), -1).sum(1))

                # gradconv_bias.append((1 / (beta * batch_size)) * (
                #         self.unpool(s[self.nc + i], inds[self.nc + i], output_size=output_size) - self.unpool(
                #     seq[self.nc + i], indseq[self.nc + i], output_size=output_size)).permute(1, 0, 2,
                #                                                                              3).contiguous().view(
                #     s[self.nc + i].size(1), -1).sum(1))
            # last layer
            output_size = [s[-1].size(0), s[-1].size(1), self.size_conv_list[-2], self.size_conv_list[-2]]
            gradW_conv.append(coef*(F.conv2d(data.permute(1,0,2,3), self.unpool(s[-1], P_ind[-1], output_size=output_size).permute(1,0,2,3), padding=self.pad)-
                                    F.conv2d(data.permute(1,0,2,3), self.unpool(seq[-1], Peq_ind[-1], output_size=output_size).permute(1,0,2,3), padding=self.pad)).permute(1,0,2,3))
            gradBias_conv.append(coef*(self.unpool(s[-1], P_ind[-1], output_size=output_size)-self.unpool(seq[-1], Peq_ind[-1], output_size=output_size)).permute(1,0,2,3).contiguous().view(s[-1].size(1),-1).sum(1))

        return gradW_conv, gradBias_conv, gradW_fc, gradBias_fc



    def updateConvWeight(self, s, seq, h, heq, P_ind, Peq_ind):

        gradW_conv, gradBias_conv, gradW_fc, gradBias_fc = self.computeConvGradientEP(self, h, heq, P_ind, Peq_ind, s, seq)

        # TODO consider change the lr sign for the random beta sign
        with torch.no_grad():
            for (fc, param) in enumerate(self.W):
                param.weight.data += self.lr[fc] * gradW_fc[fc]
                param.bias.data += self.lr[fc] * gradBias_fc[fc]

            for (cv, param) in enumerate(self.Conv):
                param.weight.data += self.lr[cv+self.fc_number] * gradW_conv[cv]
                param.weight.data += self.lr[cv+self.fc_number] * gradBias_conv[cv]

    # def initState(self, args, data):
    #     '''
    #     Inite the state of CNN
    #     '''
    #     h, s = [], []
    #     P_ind = []
    #     batch_size = data.size(0)
    #     for fc in range(len(args.fcLayers)-1):
    #         s.append(torch.zeros(batch_size, args.fcLayers[fc], requires_grad=False))
    #
    #     # TODO verify the input data size
    #     h.append(data.float())
    #     for cv in range(1, len(self.conv_number)):
    #         # data size of convolutional layer : (batch, channel, size, size)
    #         # output size is calculated by (M-F+2P)/S + 1
    #         output_size = (h[cv-1].size(-1)-args.convLayers[(cv-1)*5+2] + 2*args.convLayers[(cv-1)*5+4])/args.convLayers[(cv-1)*5+3] +1
    #         h.append(torch.zeros(batch_size, args.convLayers[(cv-1)*5+1], output_size, output_size))
    #         P_ind.append(None)
    #
    #     #  we reverse the h at the end
    #     h.reverse()
    #
    #     return s, h, P_ind

    def initHidden(self, batch_size):
        s = []
        P_ind = []
        for i in range(self.fc_number):
            s.append(torch.zeros(batch_size, self.size_classifier_tab[i], requires_grad=False)) # why we requires grad ? for comparison with BPTT?
            # inds.append(None)
        for i in range(self.conv_number):
            s.append(torch.zeros(batch_size, self.C_list[i], self.size_convpool_list[i], self.size_convpool_list[i],
                                 requires_grad=False))
            P_ind.append(None)

        return s, P_ind





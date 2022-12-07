# File defining the network and the oscillators composing the network
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from optim import rho, rhop

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
        #self.coeffDecay = args.coeffDecay
        #self.epochDecay = args.epochDecay
        self.batchSize = args.batchSize


        # define the device
        if args.device >= 0 and torch.cuda.is_available():
            device = torch.device("cuda:" + str(args.device))
            self.cuda = True
        else:
            device = torch.device("cpu")
            self.cuda = False

        self.device = device

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

    @jit.script_method
    def stepper_c_ep(self, s:List[torch.Tensor], target:Optional[torch.Tensor]=None, beta:Optional[float]=None):
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
    def forward(self, s:List[torch.Tensor], beta:Optional[float]=None, target:Optional[torch.Tensor]=None) -> List[torch.Tensor]:

        T, Kmax = self.T, self.Kmax

        with torch.no_grad():
            # continuous time EP
            if beta is None and target is not None:
                # free phase
                for t in range(T):
                    s = self.stepper_c_ep(s, target=target, beta=beta)

            else:
                # nudged phase
                for t in range(Kmax):
                    s = self.stepper_c_ep(s, target=target, beta=beta)

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

        gradW, gradBias = [], []

        with torch.no_grad():
            for layer in range(len(s)-1):
                gradW.append(coef*(torch.mm(torch.transpose(rho(s[layer+1]), 0, 1), rho(s[layer]))
                                   -torch.mm(torch.transpose(rho(seq[layer+1]), 0, 1), rho(seq[layer]))))
                gradBias.append(coef*(rho(s[layer])-rho(seq[layer])).sum(0))

        return gradW, gradBias

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
                #lrDecay = self.lr[layer]\*torch.pow(self.coeffDecay, int(epoch/self.epochDecay))
                #print('the decayed lr is:', lrDecay)
                # print('alpha is:', alpha[layer])
                lrDecay = self.lr[layer]

                self.W[layer] += lrDecay*gradW[layer]
                self.bias[layer] += lrDecay*gradBias[layer]

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

    def initHidden(self, fcLayers, data):
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

        self.batchSize = args.batchSize

        self.W = nn.ModuleList(None)

        self.Pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unPool = nn.MaxUnpool2d(2, stride=2)

        with torch.no_grad():
            for i in range(len(args.fcLayers) - 1):
                self.W.extend([nn.Linear(args.fcLayers[i + 1], args.fcLayers[i], bias=True)])
                # torch.nn.init.zeros_(self.W[i].bias)

        self.Conv = nn.ModuleList(None)
        self.P = []

        conv_number = int(len(args.convLayers) / 5)
        self.conv_number = conv_number
        self.fc_number = len(args.fcLayers) - 1

        if conv_number > 0:
            with torch.no_grad():
                for i in range(conv_number):
                    self.Conv.extend([nn.Conv2d(args.convLayers[(conv_number - i - 1) * 5],
                                                args.convLayers[(conv_number - i - 1) * 5 + 1],
                                                args.convLayers[(conv_number - i - 1) * 5 + 2],
                                                stride=args.convLayers[(conv_number - i - 1) * 5 + 3],
                                                padding=args.convLayers[(conv_number - i - 1) * 5 + 4], bias=True)])
                    self.P.append(args.convLayers[(conv_number - i - 1) * 5 + 4])

        self.unConv = nn.ModuleList(None)
        if conv_number > 1:
            with torch.no_grad():
                for i in range(conv_number - 1):
                    self.unConv.extend([nn.Conv2d(args.convLayers[(conv_number - i - 1) * 5 + 1],
                                                  args.convLayers[(conv_number - i - 1) * 5],
                                                  args.convLayers[(conv_number - i - 1) * 5 + 2],
                                                  stride=args.convLayers[(conv_number - i - 1) * 5 + 3],
                                                  padding=args.convLayers[(conv_number - i - 1) * 5 + 2] - 1 -
                                                          args.convLayers[(conv_number - i - 1) * 5 + 4], bias=False)])

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

    def stepper_p_conv(self, s, h, P_ind, target=None, beta=0):
        '''
        stepper function for prototypical convolutional model of EP
        '''

        snew = []
        snew.append(rho(self.W[0](s[1])))

        if beta != 0:
            snew[0] = snew[0] + beta*(target-s[0])

        if len(s) > 2:
            for fc in range(1, len(s)-2):
                snew.append(rho(self.W[fc](s[fc+1]) + torch.mm(s[fc-1], self.W[fc-1].weight)))

        snew.append(rho(self.W[-1](h[0].flatten(1,-1)) + torch.mm(s[-2], self.W[-2].weight)))

        hnew = []

        h0, ind = self.Pool(self.Conv[0](h[1]))
        P_ind[0] = ind
        hnew.append(rho(h0 + (torch.mm(s[-1], self.W[-1].weight)).unflatten(1, h[0].size()[1:])))
        del ind, h0

        # save the input images at h[-1], so we do not update h[-1]
        if len(h) > 2:
            for cv in range(1, len(s)-2):
                hcv, ind = self.Pool(self.Conv[cv](h[cv+1]))
                P_ind[cv] = ind
                # update the unConv weight data
                self.unConv[cv-1].weight.data = (self.Conv[cv-1].weight.flip([2, 3])).transpose(0, 1)

                hnew.append(hcv + self.unConv[cv-1](self.unPool(h[cv-1], P_ind[cv-1])))
                del hcv, ind

        # update
        s, h = snew, hnew

        return s, h, P_ind

    def forward(self, s, h, P_ind, beta=0, target=None, tracking=False):
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
                    s, h, P_ind = self.stepper_p_conv(s, h, P_ind)
                    if tracking:
                        for k in range(n_track):
                            m[k].append(s[0][k][2*k].item())
                            n[k].append(s[1][k][2*k].item())
                            p[k].append(h[0][k][2][k][k])  # out_channel=2
                            p[k].append(h[1][k][3][k][k])  # out_channel=3
            else:
                # nudging phase
                for K in range(Kmax):
                    s, h, P_ind = self.stepper_p_conv(s, h, P_ind, target=target, beta=beta)
                    if tracking:
                        for k in range(n_track):
                            m[k].append(s[0][k][2 * k].item())
                            n[k].append(s[1][k][2 * k].item())
                            p[k].append(h[0][k][2][k][k])  # out_channel=2
                            p[k].append(h[1][k][3][k][k])  # out_channel=3

        if tracking:
            return s, h, P_ind, m, n, p, q
        else:
            return s, h, P_ind

        # TODO consider C-EP (or the other methods) in the future?

    def computeConvGradientEP(self, h, heq, P_ind, Peq_ind, s, seq):
        batch_size = s[0].size(0)
        coef = 1 / (self.beta * batch_size)

        gradW_fc, gradBias_fc = [], []
        gradW_conv, gradBias_conv = [], []

        with torch.no_grad():

            if len(s) > 2:
                for fc in range(len(s)-2):
                    gradW_fc.append(coef*(torch.mm(torch.transpose(s[fc], 0, 1), s[fc+1]) -
                                         torch.mm(torch.transpose(seq[fc], 0, 1), seq[fc+1])))
                    gradBias_fc.append(coef*(s[fc]-seq[fc]).sum(0))

            # update the last layer of fc
            gradW_fc.append(coef*(torch.mm(torch.transpose(s[-1], 0, 1), h[0].flatten(1, -1)) -
                                 torch.mm(torch.transpose(seq[-1], 0, 1), heq[0].flatten(1,-1))))

            gradBias_fc.append(coef*(s[-1]-seq[-1]).sum(0))

            for cv in range(len(h)-1):
                gradW_conv.append(coef*(F.conv2d(h[cv+1].transpose(0, 1), self.unPool(h[cv], P_ind[cv]).transpose(0, 1), padding=self.P[cv]) -
                                        F.conv2d(heq[cv+1].transpose(0, 1), self.unPool(heq[cv], Peq_ind[cv]).transpose(0, 1), padding=self.P[cv])).transpose(0, 1))
                gradBias_conv.append(coef*(self.unPool(h[cv], P_ind[cv]) - self.unPool(heq[cv], Peq_ind[cv])).transpose(0, 1).sum(1, 2, 3))

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

    def initHidden(self, args, data):
        '''
        Inite the state of CNN
        '''
        h, s = [], []
        P_ind = []
        batch_size = data.size(0)
        for fc in range(len(args.fcLayers)-1):
            s.append(torch.zeros(batch_size, args.fcLayers[fc], requires_grad=False))

        # TODO verify the input data size
        h.append(data.float())
        for cv in range(1, len(self.conv_number)):
            # data size of convolutional layer : (batch, channel, size, size)
            # output size is calculated by (M-F+2P)/S + 1
            output_size = (h[cv-1].size(-1)-args.convLayers[(cv-1)*5+2] + 2*args.convLayers[(cv-1)*5+4])/args.convLayers[(cv-1)*5+3] +1
            h.append(torch.zeros(batch_size, args.convLayers[(cv-1)*5+1], output_size, output_size))
            P_ind.append(None)

        #  we reverse the h at the end
        h.reverse()

        return s, h, P_ind









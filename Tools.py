# coding: utf-8

import os
import os.path
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import*
from copy import*
import matplotlib.pyplot as plt
import sys
import pickle
import pandas as pd
import shutil
from tqdm import tqdm
import torch.nn.functional as F
from Network import*

def classify(net, jparams, class_loader):

    net.eval()

    for batch_idx, (data, targets) in enumerate(class_loader):

        # initiation of s
        s = net.initState(data)

        if net.cuda:
            targets = targets.to(net.device)#no need to put data on the GPU as data is included in s!
            for i in range(len(s)):
                s[i] = s[i].to(net.device)

        # free phase
        s = net.forward(s)

        #print('The average norm of output neurons is :', torch.norm(torch.mean(s[0], axis=0)))

        # record all the output values
        if batch_idx == 0:
            result_output = s[0].detach()
        else:
            result_output = torch.cat((result_output, s[0].detach()), 0)

        # record all the class sent
        if batch_idx == 0:
            class_vector = targets
        else:
            class_vector = torch.cat((class_vector, targets), 0)

    ##################### classifier one2one ########################

    class_moyenne = torch.zeros((jparams['n_class'], jparams['fcLayers'][0]), device=net.device)

    for i in range(jparams['n_class']):
        indice = (class_vector == i).nonzero(as_tuple=True)[0]
        result_single = result_output[indice, :]
        class_moyenne[i, :] = torch.mean(result_single, axis=0)

    # for the unclassified neurons, we kick them out from the responses
    unclassified = 0
    response = torch.argmax(class_moyenne, 0)
    # TODO to verify the difference between torch.max(output) and torch.max(class_moyenne)
    max0_indice = (torch.max(class_moyenne, 0).values == 0).nonzero(as_tuple=True)[0]
    response[max0_indice] = -1
    unclassified += max0_indice.size(0)

    return response, max0_indice

# def classify(net, jparams, class_loader):
#
#
#     net.eval()
#     class_moyenne = torch.zeros((jparams['n_class'], jparams['fcLayers'][0]), device=net.device)
#     batch_num=0
#
#     for batch_idx, (data, targets) in enumerate(class_loader):
#
#         # initiation of s
#         s = net.initState(data)
#
#         if net.cuda:
#             targets = targets.to(net.device)#no need to put data on the GPU as data is included in s!
#             for i in range(len(s)):
#                 s[i] = s[i].to(net.device)
#
#         # free phase
#         s = net.forward(s)
#         result_output = s[0].detach()
#         class_vector = targets
#
#         # calculate each class_moyenne
#         class_moyenne_batch = torch.zeros((jparams['n_class'], jparams['fcLayers'][0]), device=net.device)
#         for i in range(jparams['n_class']):
#             indice = (class_vector == i).nonzero(as_tuple=True)[0]
#             result_single = result_output[indice, :]
#             class_moyenne_batch[i, :] = torch.mean(result_single, axis=0)
#         class_moyenne += class_moyenne_batch
#         batch_num += 1
#         # # record all the output values
#         # if batch_idx == 0:
#         #     result_output = s[0].detach()
#         # else:
#         #     result_output = torch.cat((result_output, s[0].detach()), 0)
#         #
#         # # record all the class sent
#         # if batch_idx == 0:
#         #     class_vector = targets
#         # else:
#         #     class_vector = torch.cat((class_vector, targets), 0)
#
#     ##################### classifier one2one ########################
#
#     class_moyenne = class_moyenne/batch_num
#
#     for i in range(jparams['n_class']):
#         indice = (class_vector == i).nonzero(as_tuple=True)[0]
#         result_single = result_output[indice, :]
#         class_moyenne[i, :] = torch.mean(result_single, axis=0)
#
#     # for the unclassified neurons, we kick them out from the responses
#     unclassified = 0
#     response = torch.argmax(class_moyenne, 0)
#     # TODO to verify the difference between torch.max(output) and torch.max(class_moyenne)
#     max0_indice = (torch.max(class_moyenne, 0).values == 0).nonzero(as_tuple=True)[0]
#     response[max0_indice] = -1
#     unclassified += max0_indice.size(0)
#
#     return response, max0_indice


def classify_network(net, class_net, jparams, layer_loader):
    net.eval()
    class_net.train()

    # define the loss of classification layer
    if jparams['class_activation'] == 'softmax':
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.MSELoss()

    parameters = list(class_net.parameters())

    # create the list for training errors
    correct_train = torch.zeros(1, device=net.device).squeeze()
    total_train = torch.zeros(1, device=net.device).squeeze()

    # construct the optimizer
    if jparams['class_Optimizer'] == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=jparams['class_lr'])
    elif jparams['class_Optimizer'] == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=jparams['class_lr'])

    for batch_idx, (data, targets) in enumerate(layer_loader):
        optimizer.zero_grad()

        # initiation of s
        s = net.initState(data)

        if net.cuda:
            targets = targets.to(net.device)
            for i in range(len(s)):
                s[i] = s[i].to(net.device)

        # free phase
        s = net.forward(s)
        # forward propagation in classification layer
        x = s[0].clone()
        output = class_net.forward(x)
        # calculate the loss
        loss = criterion(output, targets.to(torch.float32))
        # backpropagation
        loss.backward()
        optimizer.step()

        # calculate the training errors
        prediction = torch.argmax(output, dim=1)
        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def test_unsupervised_ep_layer(net, class_net, jparams, test_loader):
    # TODO finish the test process when use

    net.eval()
    class_net.eval()

    # create the list for testing errors
    correct_test = torch.zeros(1, device=net.device).squeeze()
    total_test = torch.zeros(1, device=net.device).squeeze()
    loss_test = torch.zeros(1, device=net.device).squeeze()
    total_batch = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(test_loader):
        total_batch += 1
        s = net.initState(data)
        if net.cuda:
            targets = targets.to(net.device)
            s = [item.to(net.device) for item in s]

        # record the total test
        total_test += targets.size()[0]
        # free phase
        s = net.forward(s)
        # forward propagation in classification layer
        x = s[0].clone()
        output = class_net.forward(x)
        # calculate the loss
        if jparams['class_activation'] == 'softmax':
            loss = F.cross_entropy(output, targets)
        else:
            loss = F.mse_loss(output, F.one_hot(targets, num_classes=jparams['n_class']))

        loss_test += loss.item()

        # calculate the training errors
        prediction = torch.argmax(output, dim=1)
        correct_test += (prediction == targets).sum().float()

    # calculate the test error
    test_error = 1 - correct_test / total_test
    loss_test = loss_test / total_batch

    return test_error, loss_test


def train_supervised_crossEntropy(net, jparams, train_loader, lr, epoch):
    net.train()
    net.epoch = epoch + 1
    total_train = torch.zeros(1, device=net.device).squeeze()
    correct_train = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(train_loader):

        # random signed beta: better approximate the gradient
        net.beta = torch.sign(torch.randn(1)) * jparams['beta']
        # init the hidden layers
        h, y = net.initHidden(jparams['fcLayers'], data)

        if jparams['Dropout']:
            p_distribut, y_distribut = net.mydropout(h, p=jparams['dropProb'], y=y)
        else:
            p_distribut, y_distribut = None, None

        if net.cuda:
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)
            h = [item.to(net.device) for item in h] #no need to put data on the GPU as data is included in s!
            if jparams['Dropout']:
                p_distribut = [item.to(net.device) for item in p_distribut]
                y_distribut = y_distribut.to(net.device)

        if jparams['errorEstimate'] == 'one-sided':
            # free phase
            h, y = net.forward_softmax(h, p_distribut, y_distribut)
            heq = h.copy()
            yeq = y.clone()
            # nudging phase
            if len(h) > 1:
                h, y = net.forward_softmax(h, p_distribut, y_distribut, target=targets, beta=net.beta)
            # update the weights
            if jparams['Optimizer'] == 'Adam':
                net.Adam_updateWeight_softmax(h, heq, y, targets, lr, epoch=net.epoch)
            else:
                net.updateWeight_softmax(h, heq, y, targets, lr)

        elif jparams['errorEstimate'] == 'symmetric':
            if len(h) <= 1:
                raise ValueError("Symmetric errorEstimate will only be used for more than 1 hidden layer " "but got {} hidden layer".format(len(h)))
            # free phase
            h, y = net.forward_softmax(h, p_distribut, y_distribut)
            heq = h.copy()
            yeq = y.clone()
            # + beta
            h, y = net.forward_softmax(h, p_distribut, y_distribut, target=targets, beta=net.beta)
            hplus = h.copy()
            yplus = y.clone()
            # -beta
            h = heq.copy()
            h, y = net.forward_softmax(h, p_distribut, y_distribut, target=targets, beta=-net.beta)
            hmoins = h.copy()
            ymoins = y.clone()
        # update and track the weights of the network
            if jparams['Optimizer'] == 'Adam':
                net.Adam_updateWeight_softmax(hplus, hmoins, yplus, targets, lr, ybeta=ymoins, epoch=net.epoch)
            else:
                net.updateWeight_softmax(hplus, hmoins, yplus, targets, lr, ybeta=ymoins)

        # calculate the training error
        prediction = torch.argmax(yeq.detach(), dim=1)
        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def train_supervised_ep(net, jparams, train_loader, lr, epoch):
    net.train()
    net.epoch = epoch + 1

    total_train = torch.zeros(1, device=net.device).squeeze()
    correct_train = torch.zeros(1, device=net.device).squeeze()

    # if net.epoch % args.epochDecay == 0:
    #     net.gamma = net.gamma*args.gammaDecay
    if jparams['convNet']:
        # TODO add the dropout in the ConvNet (there is no dropout in the ConvNet yet)
        for batch_idx, (data, targets) in enumerate(train_loader):
            # random signed beta: better approximate the gradient
            net.beta = torch.sign(torch.randn(1)) * jparams['beta']

            batchSize = data.size(0)
            # initiate the neurons
            s, P_ind = net.initHidden(batchSize)

            if net.cuda:
                targets = targets.to(net.device)
                net.beta = net.beta.to(net.device)
                s = [item.to(net.device) for item in s]
                data = data.to(net.device)  # this time data is not included in the neuron list

            if jparams['errorEstimate'] == 'one-sided':
                # free phase
                s, P_ind = net.forward(s, data, P_ind)

                seq = s.copy()
                Peq_ind = P_ind.copy()

                s, P_ind = net.forward(s, data, P_ind, beta=net.beta, target=targets)

                # update weights
                net.updateConvWeight(data, s, seq, P_ind, Peq_ind, lr)

            elif jparams['errorEstimate'] == 'symmetric':
                # free phase
                s, P_ind = net.forward(s, data, P_ind)
                seq = s.copy()
                Peq_ind = P_ind.copy()

                # + beta
                s, P_ind = net.forward(s, data, P_ind, beta=net.beta, target=targets)
                splus = s.copy()
                Pplus_ind = P_ind.copy()

                s = seq.copy()
                P_ind = Peq_ind.copy()

                # -beta
                s, P_ind = net.forward(s, data, P_ind, beta=-net.beta, target=targets)
                smoins = s.copy()
                Pmoins_ind = P_ind.copy()

                # update the weights
                net.updateConvWeight(data, splus, smoins, Pplus_ind, Pmoins_ind, lr)

    else:
        for batch_idx, (data, targets) in enumerate(train_loader):

            # random signed beta: better approximate the gradient
            net.beta = torch.sign(torch.randn(1)) * jparams['beta']

            s = net.initState(data)

            if jparams['Dropout']:
                p_distribut = net.mydropout(s, p=jparams['dropProb'])
            else:
                p_distribut = None

            if net.cuda:
                targets = targets.to(net.device)
                net.beta = net.beta.to(net.device)
                s = [item.to(net.device) for item in s] #no need to put data on the GPU as data is included in s!
                if jparams['Dropout']:
                    p_distribut = [item.to(net.device) for item in p_distribut]

            if jparams['errorEstimate'] == 'one-sided':
                # free phase
                s = net.forward(s, p_distribut)
                seq = s.copy()
                s = net.forward(s, p_distribut, target=targets, beta=net.beta)

                if jparams['Optimizer'] == 'Adam':
                    net.Adam_updateWeight(s, seq, lr, epoch=net.epoch)
                else:
                    net.updateWeight(s, seq, lr)
            elif jparams['errorEstimate'] == 'symmetric':
                # free phase
                s = net.forward(s, p_distribut)
                seq = s.copy()
                # + beta
                s = net.forward(s, p_distribut, target=targets, beta=net.beta)
                splus = s.copy()
                # -beta
                s = seq.copy()
                s = net.forward(s, p_distribut, target=targets, beta=-net.beta)
                smoins = s.copy()
            # update and track the weights of the network
                if jparams['Optimizer'] == 'Adam':
                    net.Adam_updateWeight(splus, smoins, lr, epoch=net.epoch)
                else:
                    net.updateWeight(splus, smoins, lr)

    # calculate the training error
    prediction = torch.argmax(seq[0].detach(), dim=1)
    correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
    total_train += targets.size(dim=0)

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


# TODO the function of unsupervised_crossEntropy can be used in semi-supervised learning
# TODO add the dropout
def train_unsupervised_crossEntropy(net, jparams, train_loader, lr, epoch):
    net.train()
    net.epoch = epoch + 1

    # Stochastic mode
    if jparams['batchSize'] == 1:
        Y_p = torch.zeros(jparams['fcLayers'][0], device=net.device)

    Xth = torch.zeros(jparams['fcLayers'][0], device=net.device)
    # Decide the decreasing gamma
    if net.epoch % jparams['epochDecay'] == 0:
        net.gamma = net.gamma*jparams['gammaDecay']

    # decide the factor
    T_coef = 0.002
    target_activity = jparams['nudge_N'] / jparams['fcLayers'][0]

    for batch_idx, (data, targets) in enumerate(train_loader):

        # random signed beta: better approximate the gradient
        net.beta = torch.sign(torch.randn(1)) * jparams['beta']
        # init the hidden layers
        h, y = net.initHidden(jparams['fcLayers'], data)

        if jparams['Dropout']:
            p_distribut, y_distribut = net.mydropout(h, p=jparams['dropProb'], y=y)
        else:
            p_distribut, y_distribut = None, None

        if net.cuda:
            targets = targets.to(net.device)  # targets here were not encoded by one-hot coding
            net.beta = net.beta.to(net.device)
            h = [item.to(net.device) for item in h] #no need to put data on the GPU as data is included in s!
            if jparams['Dropout']:
                p_distribut = [item.to(net.device) for item in p_distribut]
                y_distribut = y_distribut.to(net.device)

        if jparams['errorEstimate'] == 'one-sided':
            # free phase
            h, y = net.forward_softmax(h, p_distribut, y_distribut)
            heq = h.copy()
            yeq = y.clone()
            # # define the targets by creating a new softmax function
            # unsupervised_targets = F.softmax(yeq/T_coef, dim=1)
            # define the targets by argmax
            unsupervised_targets, maxindex = net.unsupervised_target(yeq, jparams['nudge_N'], Xth)
            # nudging phase
            if len(h) > 1:
                h, y = net.forward_softmax(h, p_distribut, y_distribut, target=unsupervised_targets, beta=net.beta)

            # update the weights
            if jparams['Optimizer'] == 'Adam':
                net.Adam_updateWeight_softmax(h, heq, y, unsupervised_targets, lr, epoch=net.epoch)
            else:
                net.updateWeight_softmax(h, heq, y, unsupervised_targets, lr)

        elif jparams['errorEstimate'] == 'symmetric':
            if len(h) <= 1:
                raise ValueError("Symmetric errorEstimate will only be used for more than 1 hidden layer " "but got {} hidden layer".format(len(h)))
            # free phase
            h, y = net.forward_softmax(h, p_distribut, y_distribut)
            heq = h.copy()
            yeq = y.clone()
            # # define the unsupervised targets
            # unsupervised_targets = F.softmax(yeq/T_coef, dim=1)
            # define the unsupervised targets by argmax
            unsupervised_targets, maxindex = net.unsupervised_target(yeq, jparams['nudge_N'], Xth)
            # + beta
            h, y = net.forward_softmax(h, p_distribut, y_distribut, target=unsupervised_targets, beta=net.beta)
            hplus = h.copy()
            yplus = y.clone()
            # -beta
            h = heq.copy()
            h, y = net.forward_softmax(h, p_distribut, y_distribut, target=unsupervised_targets, beta=-net.beta)
            hmoins = h.copy()
            ymoins = y.clone()
        # update and track the weights of the network
            if jparams['Optimizer'] == 'Adam':
                net.Adam_updateWeight_softmax(hplus, hmoins, yplus, unsupervised_targets, lr, ybeta=ymoins, epoch=net.epoch)
            else:
                net.updateWeight_softmax(hplus, hmoins, yplus, unsupervised_targets, lr, ybeta=ymoins)
        # calculate the Homeostasis
        # nudge_sign = torch.sign(unsupervised_targets-yeq)
        # A = torch.max(nudge_sign, torch.zeros(nudge_sign.size(), device=net.device))

        # if args.batchSize == 1:
        #     Y_p = (1 - args.eta) * Y_p + args.eta * A[0]
        #     Xth += net.gamma * (Y_p - target_activity)
        # else:
        #     Xth += net.gamma * (torch.mean(A, axis=0) - target_activity)
        if jparams['batchSize'] == 1:
            Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * unsupervised_targets[0]
            Xth += net.gamma * (Y_p - target_activity)
        else:
            Xth += net.gamma * (torch.mean(unsupervised_targets, axis=0) - target_activity)

    return Xth


def train_unsupervised_ep(net, jparams, train_loader, lr, epoch):
    '''
    Function to train the network for 1 epoch
    '''
    #TODO there is no Adam no symmetric update in unsupervised

    net.train()
    net.epoch = epoch+1
    # Stochastic mode
    if jparams['batchSize'] == 1:
        Y_p = torch.zeros(jparams['fcLayers'][0], device=net.device)

    Xth = torch.zeros(jparams['fcLayers'][0], device=net.device)
    # Decide the decreasing gamma
    if net.epoch % jparams['epochDecay'] == 0:
        net.gamma = net.gamma*jparams['gammaDecay']

    for batch_idx, (data, targets) in enumerate(train_loader):
        #random signed beta: better approximate the gradient
        net.beta = torch.sign(torch.randn(1)) * jparams['beta']

        #net.beta = args.beta

        s = net.initState(data)

        if jparams['Dropout']:
            p_distribut = net.mydropout(s, p=jparams['dropProb'])
        else:
            p_distribut = None
        # print('This distribut is:', p_distribut)

        if net.cuda:
            net.beta = net.beta.to(net.device)
            s = [item.to(net.device) for item in s] #no need to put data on the GPU as data is included in s!
            if jparams['Dropout']:
                p_distribut = [item.to(net.device) for item in p_distribut]

        # weight normalization before the free phase
        if jparams['weightNormalization']:
            net.weightNormalization()

        if jparams['errorEstimate'] == 'one-sided':
            # free phase
            s = net.forward(s, p_distribut)
            seq = s.copy()

            # unsupervised targets
            output = s[0].clone()
            unsupervised_targets, maxindex = net.unsupervised_target(output, jparams['nudge_N'], Xth)

            # nudging phase
            s = net.forward(s, p_distribut, target=unsupervised_targets, beta=net.beta)

            # update the weights
            if jparams['Optimizer'] == 'Adam':
                net.Adam_updateWeight(s, seq, lr, epoch=net.epoch)
            else:
                net.updateWeight(s, seq, lr, epoch=net.epoch)

        elif jparams['errorEstimate'] == 'symmetric':

            # free phase
            s = net.forward(s, p_distribut)
            seq = s.copy()

            # unsupervised target
            output = s[0].clone()
            unsupervised_targets, maxindex = net.unsupervised_target(output, jparams['nudge_N'], Xth)

            # + beta
            s = net.forward(s, p_distribut, target=unsupervised_targets, beta=net.beta)
            splus = s.copy()

            # - beta
            s = seq.copy()
            s = net.forward(s, p_distribut, target=unsupervised_targets, beta=-net.beta)
            smoins = s.copy()

            # update and track the weights of the network
            if jparams['Optimizer'] == 'Adam':
                net.Adam_updateWeight(splus, smoins, lr, epoch=net.epoch)
            else:
                net.updateWeight(splus, smoins, lr, epoch=net.epoch)

        if jparams['Dropout']:
            target_activity =jparams['nudge_N'] / (jparams['fcLayers'][0] * (1 - jparams['dropProb'][0]))  # dropout influences the target activity
            if jparams['batchSize'] == 1:
                Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * unsupervised_targets[0]

                Xth += net.gamma * (Y_p - target_activity)*p_distribut[0]
            else:
                Xth += net.gamma * ((torch.sum(unsupervised_targets, axis=0)/torch.sum(p_distribut[0], axis=0)) -target_activity)

        else:
            target_activity = jparams['nudge_N'] / jparams['fcLayers'][0]
            if jparams['batchSize'] == 1:
                Y_p = (1 - jparams['eta']) * Y_p + jparams['eta'] * unsupervised_targets[0]
                Xth += net.gamma * (Y_p - target_activity)
            else:
                Xth += net.gamma * (torch.mean(unsupervised_targets, axis=0) - target_activity)
                #Xth += net.gamma * (torch.sum(unsupervised_targets, axis=0)/torch.sum(p_distribut, axis=0)) -

    # TODO make a version compatible with SGD and Adam
    return Xth


def test_unsupervised_ep(net, jparams, test_loader, response, record=None):
    '''
        Function to test the network
        '''
    net.eval()

    criterion = nn.MSELoss(reduction='sum')  # what it for?????

    # record total test number
    total_test = torch.zeros(1, device=net.device).squeeze()

    # record unsupervised test error
    correct_av_test = torch.zeros(1, device=net.device).squeeze()
    correct_max_test = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(test_loader):

        s = net.initState(data)

        if net.cuda:
            targets = targets.to(net.device)
            s = [item.to(net.device) for item in s]

        # record the total test
        total_test += targets.size()[0]

        # free phase
        s = net.forward(s)

        # we note the last layer as s_output
        output = s[0].clone()

        '''average value'''
        classvalue = torch.zeros(output.size(0), jparams['n_class'], device=net.device)

        for i in range(jparams['n_class']):
            indice = (response == i).nonzero(as_tuple=True)[0]
            # TODO need to consider the situation that one class is not presented
            if len(indice) == 0:
                classvalue[:, i] = -1
            else:
                classvalue[:, i] = torch.mean(output[:, indice], 1)

        predict_av = torch.argmax(classvalue, 1)
        correct_av_test += (predict_av == targets).sum().float()

        '''maximum value'''
        # remove the non response neurons
        non_response_indice = (response == -1).nonzero(as_tuple=True)[0]
        output[:, non_response_indice] = -1

        maxindex_output = torch.argmax(output, 1)
        predict_max = response[maxindex_output]
        correct_max_test += (predict_max == targets).sum().float()

        '''record the average and max classification'''
        if record is not None:
            if batch_idx == 0:
                av_record = predict_av
                max_record = predict_max
            else:
                av_record = torch.cat((av_record, predict_av), 0)
                max_record = torch.cat((max_record, predict_max), 0)

    # calculate the test error
    test_error_av = 1 - correct_av_test / total_test
    test_error_max = 1 - correct_max_test / total_test

    if record is not None:
        return test_error_av, test_error_max, av_record, max_record
    else:
        return test_error_av, test_error_max


def test_supervised_ep(net, test_loader, record=None):
    '''
    Function to test the network
    '''
    net.eval()

    criterion = nn.MSELoss(reduction = 'sum') #???

    # record total test number
    total_test = torch.zeros(1, device=net.device).squeeze()

    # record supervised test error
    corrects_supervised = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(test_loader):

        s = net.initState(data)

        if net.cuda:
            targets = targets.to(net.device)
            s = [item.to(net.device) for item in s]

        # record the total test
        total_test += targets.size()[0]

        #free phase
        s = net.forward(s)

        # we note the last layer as s_output
        output = s[0].clone().detach()

        #
        prediction = torch.argmax(output, dim=1)
        corrects_supervised += (prediction == targets).sum().float()

        if record is not None:
            if batch_idx == 0:
                test_class_record = prediction
            else:
                test_class_record = torch.cat((test_class_record, prediction), 0)

    test_error = 1 - corrects_supervised / total_test

    if record is not None:
        return test_error, test_class_record
    else:
        return test_error


def initDataframe(path, method='supervised', dataframe_to_init='results.csv'):
    '''
    Initialize a dataframe with Pandas so that parameters are saved
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep = ',', index_col = 0)
    else:
        if method == 'supervised':
            columns_header = ['Train_Error', 'Min_Train_Error', 'Test_Error', 'Min_Test_Error']
        elif method == 'unsupervised':
            columns_header = ['Test_Error_av', 'Min_Test_Error_av', 'Test_Error_max', 'Min_Test_Error_max']
        elif method == 'semi-supervised':
            columns_header = ['Supervised_Test_Error', 'Min_Supervised_Test_Error', 'Entire_Test_Error', 'Min_Entire_Test_Error']
        elif method == 'classification_layer':
            columns_header = ['Train_Class_Error', 'Min_Train_Class_Error', 'Final_Test_Error', 'Min_Final_Test_Error',
                              'Final_Test_Loss', 'Min_Final_Test_Loss']
        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(path + prefix + dataframe_to_init)
    return dataframe


def initXthframe(path, dataframe_to_init='Xth_norm.csv'):
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    if os.path.isfile(path + dataframe_to_init):
        dataframe = pd.read_csv(path + dataframe_to_init, sep=',', index_col=0)
    else:
        columns_header = ['Xth_norm']

        dataframe = pd.DataFrame({}, columns=columns_header)
        dataframe.to_csv(path + prefix + 'Xth_norm.csv')
    return dataframe


def updateDataframe(BASE_PATH, dataframe, error1, error2, filename='results.csv', loss=None):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'
    if loss is None:
        data = [error1[-1], min(error1), error2[-1], min(error2)]
    else:
        data = [error1[-1], min(error1), error2[-1], min(error2), loss[-1], min(loss)]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(BASE_PATH + prefix + filename)
    except PermissionError:
        input("Close the "+filename+" and press any key.")

    return dataframe


def updateXthframe(BASE_PATH, dataframe, Xth_norm):

    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    data = [Xth_norm[-1]]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(BASE_PATH + prefix + 'Xth_norm.csv')
    except PermissionError:
        input("Close the Xth_norm.csv and press any key.")

    return dataframe


def createPath(args=None):
    '''
    Create path to save data
    '''
    if os.name != 'posix':
        BASE_PATH = "\\\\?\\" + os.getcwd()
        prefix = '\\'

    else:
        BASE_PATH = os.getcwd()
        prefix = '/'

    BASE_PATH +=  prefix + 'DATA-0'

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


# this function will not be used since now we use json file instead
def saveHyperparameters(args, BASE_PATH):
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

    for key in args.__dict__:
        f.write(key)
        f.write(': ')
        f.write(str(args.__dict__[key]))
        f.write('\n')

    f.close()

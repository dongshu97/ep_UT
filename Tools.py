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

from Network import*


def classify(net, args, class_loader):

    net.eval()

    for batch_idx, (data, targets) in enumerate(class_loader):

        # initiation of s
        s = net.initHidden(args.fcLayers, data)

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

    class_moyenne = torch.zeros((args.n_class, args.fcLayers[0]), device=net.device)

    for i in range(args.n_class):
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


def train_supervised_ep(net, args, train_loader, epoch):
    net.train()
    net.epoch = epoch + 1

    total_train = torch.zeros(1, device=net.device).squeeze()
    correct_train = torch.zeros(1, device=net.device).squeeze()

    # if net.epoch % args.epochDecay == 0:
    #     net.gamma = net.gamma*args.gammaDecay

    for batch_idx, (data, targets) in enumerate(train_loader):

        # random signed beta: better approximate the gradient
        net.beta = torch.sign(torch.randn(1)) * args.beta

        s = net.initHidden(args.fcLayers, data)

        if net.cuda:
            targets = targets.to(net.device)
            net.beta = net.beta.to(net.device)
            s = [item.to(net.device) for item in s] #no need to put data on the GPU as data is included in s!

        #free phase
        s = net.forward(s)

        seq = s.copy()

        s = net.forward(s, target=targets, beta=net.beta)

        # update and track the weights of the network
        #net.updateWeight(s, seq)
        if args.Optimizer=='Adam':
            net.Adam_updateWeight(s, seq, epoch=net.epoch)
        else:
            net.updateWeight(s, seq)

        # calculate the training error
        prediction = torch.argmax(seq[0].detach(), dim=1)
        correct_train += (prediction == torch.argmax(targets, dim=1)).sum().float()
        total_train += targets.size(dim=0)

    # calculate the train error
    train_error = 1 - correct_train / total_train
    return train_error


def train_unsupervised_ep(net, args, train_loader, epoch, mW=[0,0], vW=[0,0], mBias=[0,0], vBias=[0,0]):
    '''
    Function to train the network for 1 epoch
    '''
    net.train()
    net.epoch = epoch+1
    # Stochastic mode
    if args.batchSize == 1:
        Y_p = torch.zeros(args.fcLayers[0], device=net.device)

    Xth = torch.zeros(args.fcLayers[0], device=net.device)
    # Decide the decreasing gamma
    if net.epoch % args.epochDecay == 0:
        net.gamma = net.gamma*args.gammaDecay

    # if args.Dropout:
    #     myDropout = Dropout()

    for batch_idx, (data, targets) in enumerate(train_loader):

        #random signed beta: better approximate the gradient
        net.beta = torch.sign(torch.randn(1)) * args.beta

        #net.beta = args.beta

        s = net.initHidden(args.fcLayers, data)

        if net.cuda:
            net.beta = net.beta.to(net.device)
            s = [item.to(net.device) for item in s] #no need to put data on the GPU as data is included in s!

        # weight normalization before the free phase
        if args.weightNormalization:
            net.weightNormalization()

        if args.Dropout:
            p_distribut = net.mydropout(s, p=args.dropProb)
            #print('This distribut is:', p_distribut)
            if net.cuda:
                p_distribut = [item.to(net.device) for item in p_distribut]

            # free phase
            s = net.forward(s, p_distribut)

            seq = s.copy()

            # unsupervised target
            output = s[0].clone()

            unsupervised_targets, maxindex = net.unsupervised_target(output, args.nudge_N, Xth)

            # nudging phase
            s = net.forward(s, p_distribut, target=unsupervised_targets, beta=net.beta)

        else:
            # free phase
            s = net.forward(s)
            seq = s.copy()

            # unsupervised target
            output = s[0].clone()

            unsupervised_targets, maxindex = net.unsupervised_target(output, args.nudge_N, Xth)

            #print('unsupervised_targets is:', unsupervised_targets)
            #print('maxindex is:', maxindex)

            # nudging phase
            s = net.forward(s, target=unsupervised_targets, beta=net.beta)

        # update and track the weights of the network
        net.updateWeight(s, seq, epoch=epoch+1)
        # elif args.Optimizer == 'Adam':
        #     mW, vW, mBias, vBias = net.updateWeight(s, seq, epoch+1, mW, vW, mBias, vBias, args.lr)

        # update homeostasis term Xth
        target_activity = args.nudge_N / (args.fcLayers[0]*(1-args.dropProb[0]))
        if args.Dropout:
            if args.batchSize == 1:
                Y_p = (1 - args.eta) * Y_p + args.eta * unsupervised_targets[0]

                Xth += net.gamma * (Y_p - target_activity)*p_distribut[0]
            else:
                Xth += net.gamma * ((torch.sum(unsupervised_targets, axis=0)/torch.sum(p_distribut[0], axis=0)) -target_activity)

        else:
            if args.batchSize == 1:
                Y_p = (1 - args.eta) * Y_p + args.eta * unsupervised_targets[0]
                Xth += net.gamma * (Y_p - target_activity)
            else:
                Xth += net.gamma * (torch.mean(unsupervised_targets, axis=0) - target_activity)
                #Xth += net.gamma * (torch.sum(unsupervised_targets, axis=0)/torch.sum(p_distribut, axis=0)) -

    # TODO make a version compatible with SGD and Adam
    return Xth


def test_unsupervised_ep(net, args, test_loader, response):
    '''
        Function to test the network
        '''
    net.eval()

    criterion = nn.MSELoss(reduction='sum')  # ???

    # record total test number
    total_test = torch.zeros(1, device=net.device).squeeze()

    # record unsupervised test error
    correct_av_test = torch.zeros(1, device=net.device).squeeze()
    correct_max_test = torch.zeros(1, device=net.device).squeeze()

    for batch_idx, (data, targets) in enumerate(test_loader):

        s = net.initHidden(args.fcLayers, data)

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
        classvalue = torch.zeros(output.size(0), args.n_class, device=net.device)

        for i in range(args.n_class):
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

    # calculate the test error
    test_error_av = 1 - correct_av_test / total_test
    test_error_max = 1 - correct_max_test / total_test

    return test_error_av, test_error_max


def test_supervised_ep(net, args, test_loader):
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

        s = net.initHidden(args.fcLayers, data)

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

    test_error = 1 - corrects_supervised / total_test
    return test_error


def initDataframe(path, args, net, method='supervised', dataframe_to_init = 'results.csv'):
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

        dataframe = pd.DataFrame({}, columns = columns_header)
        dataframe.to_csv(path + prefix + 'results.csv')
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


def updateDataframe(BASE_PATH, dataframe, error1, error2):
    '''
    Add data to the pandas dataframe
    '''
    if os.name != 'posix':
        prefix = '\\'
    else:
        prefix = '/'

    data = [error1[-1], min(error1), error2[-1], min(error2)]

    new_data = pd.DataFrame([data], index=[1], columns=dataframe.columns)

    dataframe = pd.concat([dataframe, new_data], axis=0)

    try:
        dataframe.to_csv(BASE_PATH + prefix + 'results.csv')
    except PermissionError:
        input("Close the results.csv and press any key.")

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


def createPath(args):
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


def saveHyperparameters(args, net, BASE_PATH):
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

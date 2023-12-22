import optuna
from Tools import *
from tqdm import tqdm

if os.name != 'posix':
    prefix = '\\'
else:
    prefix = '/'


def defineOptimizer(net, lr, type, convNet=False, momentum=0, dampening=0):
# convNet is a bool value
    net_params = []
    if convNet:
        for i in range(len(net.W)):
            net_params += [{'params': net.W[i].weight, 'lr':lr[i]}]
            net_params += [{'params': net.W[i].bias, 'lr':lr[i]}]
        for i in range(len(net.Conv)):
            net_params += [{'params': net.Conv[i].weight, 'lr': lr[i+len(net.W)]}]
            net_params += [{'params': net.Conv[i].bias, 'lr': lr[i+len(net.W)]}]
    else:
        for i in range(len(net.W)):
            net_params += [{'params': [net.W[i]], 'lr': lr[i]}]
            net_params += [{'params': [net.bias[i]], 'lr': lr[i]}]
    if type == 'SGD':
        optimizer = torch.optim.SGD(net_params, momentum=momentum, dampening=dampening)
    elif type == 'Adam':
        optimizer = torch.optim.Adam(net_params)
    else:
        raise ValueError("{} type of Optimizer is not defined ".format(type))

    return net_params, optimizer

def defineOptimizer_classlayer(net, lr, type, momentum=0, dampening=0):

    # define Optimizer
    layer_names = []
    for idx, (name, param) in enumerate(net.named_parameters()):
        layer_names.append(name)
    parameters = []

    for idx, name in enumerate(layer_names):
        # update learning rate
        if idx % 2 == 0:
            lr_indx = int(idx / 2)
            lr_layer = lr[lr_indx]
        # append layer parameters
        parameters += [{'params': [p for n, p in net.named_parameters() if n == name and p.requires_grad],
                        'lr': lr_layer}]

    # construct the optimizer
    # TODO changer optimizer to ADAM
    if type == 'SGD':
        optimizer = torch.optim.SGD(parameters, momentum=momentum, dampening=dampening)
    elif type == 'Adam':
        optimizer = torch.optim.Adam(parameters)
    return parameters, optimizer

def defineScheduler(optimizer, type, decay_factor, decay_epoch, exponential_factor):
    # linear
    if type == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1,
                                                           end_factor=decay_factor,
                                                           total_iters=decay_epoch)
    # exponential
    elif type == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exponential_factor)
    # step
    elif type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=decay_epoch, gamma=decay_factor)
    # combine cosine
    elif type == 'cosine':
        scheduler1 = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=decay_factor,
                                                         total_iters=decay_epoch)
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=decay_epoch)
        scheduler = torch.optim.lr_scheduler.ChainedScheduler([scheduler1, scheduler2])

    return scheduler


def supervised_ep(net, jparams, train_loader, test_loader, BASE_PATH=None, trial=None):

    # define optimizer
    params, optimizer = defineOptimizer(net, jparams['lr'], jparams['Optimizer'], convNet=jparams['convNet'])

    # TODO define scheduler

    if BASE_PATH is not None:
        DATAFRAME = initDataframe(BASE_PATH, method='supervised')

        if jparams['convNet']:
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict_0.pt')
        else:
            with open(BASE_PATH + prefix + 'model_initial.pt', 'wb') as f:
                torch.jit.save(net, f)

        train_error_list = []
        test_error_list = []

    for epoch in tqdm(range(jparams['epochs'])):
        # train process
        if jparams['lossFunction'] == 'MSE':
            train_error_epoch = train_supervised_ep(net, jparams, train_loader, optimizer, epoch)
        elif jparams['lossFunction'] == 'Cross-entropy':
            if jparams['convNet']:
                raise ValueError("convNet can not be integrated with Cross-entropy yet")
            train_error_epoch = train_supervised_crossEntropy(net, jparams, train_loader, optimizer, epoch)

        # test process
        test_error_epoch = test_supervised_ep(net, jparams, test_loader, jparams['lossFunction'])

        # TODO scheduler step

        # add optuna pruning process
        if trial is not None:
            trial.report(test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            train_error_list.append(train_error_epoch.item())
            test_error_list.append(test_error_epoch.item())

            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, train_error_list, test_error_list)
            # save the inference model
            # torch.save(net.state_dict(), BASE_PATH)

            # save the entire model
            if jparams['convNet']:
                torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict_entire.pt')
            else:
                with open(BASE_PATH + prefix + 'model_entire.pt', 'wb') as f:
                    torch.jit.save(net, f)
    if trial is not None:
        return test_error_epoch


def unsupervised_ep(net, jparams, train_loader, class_loader, test_loader, layer_loader, BASE_PATH=None, trial=None):

    # define optimizer
    params, optimizer = defineOptimizer(net, jparams['lr'], jparams['Optimizer'], convNet=jparams['convNet'])

    # define scheduler
    scheduler = defineScheduler(optimizer, jparams['scheduler'], jparams['factor'], jparams['scheduler_epoch'], jparams['exp_factor'])

    if BASE_PATH is not None:
        DATAFRAME = initDataframe(BASE_PATH, method='unsupervised')

        # dataframe for Xth
        Xth_dataframe = initXthframe(BASE_PATH, 'Xth_norm.csv')

        # save the initial network
        if jparams['convNet']:
            torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict_0.pt')
        else:
            with open(BASE_PATH + prefix + 'model_initial.pt', 'wb') as f:
                torch.jit.save(net, f)

        test_error_list_av = []
        test_error_list_max = []

        Xth_record = []

    for epoch in tqdm(range(jparams['epochs'])):
        # train process
        if jparams['lossFunction'] == 'MSE':
            Xth = train_unsupervised_ep(net, jparams, train_loader, optimizer, epoch)
        elif jparams['lossFunction'] == 'Cross-entropy':
            Xth = train_unsupervised_crossEntropy(net, jparams, train_loader, optimizer, epoch)

        # one2one class process
        response = classify(net, jparams, class_loader)

        # test process
        error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, test_loader, response)

        # scheduler
        scheduler.step()

        # add optuna pruning process
        if trial is not None:
            trial.report(error_av_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            test_error_list_av.append(error_av_epoch.item())
            test_error_list_max.append(error_max_epoch.item())
            Xth_record.append(torch.norm(Xth).item())

            DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, test_error_list_av, test_error_list_max)
            Xth_dataframe = updateXthframe(BASE_PATH, Xth_dataframe, Xth_record)

            # save the entire model
            if jparams['convNet']:
                torch.save(net.state_dict(), BASE_PATH + prefix + 'model_state_dict_entire.pt')
            else:
                with open(BASE_PATH + prefix + 'model_entire.pt', 'wb') as f:
                    torch.jit.save(net, f)

    if trial is not None:
        return error_av_epoch

    # final one2one
    response = classify(net, jparams, class_loader)
    error_av_epoch, error_max_epoch = test_unsupervised_ep(net, jparams, test_loader, response)
    if jparams['epochs']==0 and BASE_PATH is not None:
        test_error_list_av.append(error_av_epoch.item())
        test_error_list_max.append(error_max_epoch.item())
        DATAFRAME = updateDataframe(BASE_PATH, DATAFRAME, test_error_list_av, test_error_list_max)

    # we create the layer for classfication
    train_class_layer(net, jparams, layer_loader, test_loader, trained_path=None, BASE_PATH=BASE_PATH, trial=None)


def semi_supervised_ep(net, jparams, supervised_loader, unsupervised_loader, test_loader,
                       trained_path=None, BASE_PATH=None, trial=None):
    if trained_path is not None:
        with open(trained_path, 'rb') as f:
                loaded_net = torch.jit.load(f, map_location=net.device)
                net.W = loaded_net.W.copy()
                net.bias = loaded_net.bias.copy()
    else:
        # pre-train
        pre_supervised_ep(net, jparams, supervised_loader, test_loader, BASE_PATH=BASE_PATH)

    # initial pretrain error
    initial_pretrain_err = test_supervised_ep(net, jparams, test_loader, jparams['lossFunction'])

    # define the supervised and unsupervised optimizer
    unsupervised_params, unsupervised_optimizer = defineOptimizer(net, jparams['lr'], jparams['Optimizer'])
    supervised_params, supervised_optimizer = defineOptimizer(net, jparams['lr'], jparams['Optimizer'])

    # define the supervised and unsupervised scheduler
    unsupervised_scheduler = torch.optim.lr_scheduler.LinearLR(unsupervised_optimizer,
                                                               start_factor=jparams['unsupervised_start'],
                                                               end_factor=jparams['unsupervised_end'], total_iters=jparams['epochs'])
    # TODO to be verify
    supervised_scheduler = torch.optim.lr_scheduler.LinearLR(supervised_optimizer,
                                                             start_factor=jparams['supervised_start'],
                                                             end_factor=jparams['supervised_end'], total_iters=jparams['epochs'])

    if BASE_PATH is not None:
        SEMIFRAME = initDataframe(BASE_PATH, method='semi-supervised', dataframe_to_init='semi-supervised.csv')

        supervised_test_error_list = []
        entire_test_error_list = []

    for epoch in tqdm(range(jparams['epochs'])):
        # unsupervised train
        if jparams['lossFunction'] == 'MSE':
            Xth = train_unsupervised_ep(net, jparams, unsupervised_loader, unsupervised_optimizer, epoch)
        elif jparams['lossFunction'] == 'Cross-entropy':
            Xth = train_unsupervised_crossEntropy(net, jparams, unsupervised_loader, unsupervised_optimizer, epoch)
        # unsupervised test
        entire_test_epoch = test_supervised_ep(net, jparams, test_loader, jparams['lossFunction'])
        # unsupervised scheduler
        unsupervised_scheduler.step()

        # supervised train
        if jparams['lossFunction'] == 'MSE':
            pretrain_error_epoch = train_supervised_ep(net, jparams, supervised_loader, supervised_optimizer, epoch)
        elif jparams['lossFunction'] == 'Cross-entropy':
            pretrain_error_epoch = train_supervised_crossEntropy(net, jparams, supervised_loader, supervised_optimizer,
                                                                 epoch)
        # supervised test
        supervised_test_epoch = test_supervised_ep(net, jparams, test_loader, jparams['lossFunction'])
        # supervised scheduler
        supervised_scheduler.step()

        if trial is not None:
            trial.report(supervised_test_epoch, epoch)
            if entire_test_epoch > initial_pretrain_err:
                raise optuna.TrialPruned()
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            supervised_test_error_list.append(supervised_test_epoch.item())
            entire_test_error_list.append(entire_test_epoch.item())
            SEMIFRAME = updateDataframe(BASE_PATH, SEMIFRAME, entire_test_error_list, supervised_test_error_list,
                                        'semi-supervised.csv')
            with open(BASE_PATH + prefix + 'model_semi_entire.pt', 'wb') as f:
                torch.jit.save(net, f)
    if trial is not None:
        return supervised_test_epoch


def pre_supervised_ep(net, jparams, supervised_loader, test_loader, BASE_PATH=None, trial=None):

    # define pre_train optimizer
    pretrain_params, pretrain_optimizer = defineOptimizer(net, jparams['pre_lr'], jparams['Optimizer'])

    # define pre_train scheduler
    pretrain_scheduler = defineScheduler(pretrain_optimizer, jparams['pre_scheduler'], jparams['pre_factor'],
                                         jparams['pre_scheduler_epoch'], jparams['pre_exp_factor'])

    if BASE_PATH is not None:
        PretrainFrame = initDataframe(BASE_PATH, method='supervised', dataframe_to_init='pre_supervised.csv')

        # save the initial network
        with open(BASE_PATH + prefix + 'model_pre_supervised0.pt', 'wb') as f:
            torch.jit.save(net, f)

        pretrain_error_list = []
        pretest_error_list = []

    # TODO load the pretrained network for optuna

    for epoch in tqdm(range(jparams['pre_epochs'])):
        # train
        if jparams['lossFunction'] == 'MSE':
            pretrain_error_epoch = train_supervised_ep(net, jparams, supervised_loader, pretrain_optimizer, epoch)
        elif jparams['lossFunction'] == 'Cross-entropy':
            pretrain_error_epoch = train_supervised_crossEntropy(net, jparams, supervised_loader, pretrain_optimizer,
                                                                 epoch)
        # test
        pretest_error_epoch = test_supervised_ep(net, jparams, test_loader, jparams['lossFunction'])
        # scheduler
        pretrain_scheduler.step()

        if trial is not None:
            trial.report(pretest_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            pretrain_error_list.append(pretrain_error_epoch.item())
            pretest_error_list.append(pretest_error_epoch.item())
            PretrainFrame = updateDataframe(BASE_PATH, PretrainFrame, pretrain_error_list, pretest_error_list,
                                            'pre_supervised.csv')
            # save the entire model
            with open(BASE_PATH + prefix + 'model_pre_supervised_entire.pt', 'wb') as f:
                torch.jit.save(net, f)

    if trial is not None:
        return pretest_error_epoch


def train_class_layer(net, jparams, layer_loader, test_loader, trained_path=None, BASE_PATH=None, trial=None):
    # load the pre-trianed network
    if trained_path is not None:
        with open(trained_path, 'rb') as f:
            loaded_net = torch.jit.load(f, map_location=net.device)
            net.W = loaded_net.W.copy()
            net.bias = loaded_net.bias.copy()

    # create the classification layer
    class_net = Classifier(jparams)

    # define optimizer
    class_params, class_optimizer = defineOptimizer_classlayer(class_net, jparams['class_lr'], jparams['class_Optimizer'])

    # define scheduler
    class_scheduler = defineScheduler(class_optimizer, jparams['class_scheduler'], jparams['class_factor'],
                                jparams['class_scheduler_epoch'], jparams['class_exp_factor'])

    if BASE_PATH is not None:
        # create dataframe for classification layer
        class_dataframe = initDataframe(BASE_PATH, method='classification_layer',
                                        dataframe_to_init='classification_layer.csv')
        torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict_0.pt')
        class_train_error_list = []
        final_test_error_list = []
        final_loss_error_list = []

    for epoch in tqdm(range(jparams['class_epoch'])):
        # train
        class_train_error_epoch = classify_network(net, class_net, jparams, layer_loader, class_optimizer, jparams['class_smooth'])
        # test
        final_test_error_epoch, final_loss_epoch = test_unsupervised_ep_layer(net, class_net, jparams, test_loader)
        # scheduler
        class_scheduler.step()

        if trial is not None:
            trial.report(final_test_error_epoch, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if BASE_PATH is not None:
            class_train_error_list.append(class_train_error_epoch.item())
            final_test_error_list.append(final_test_error_epoch.item())
            final_loss_error_list.append(final_loss_epoch.item())
            class_dataframe = updateDataframe(BASE_PATH, class_dataframe, class_train_error_list, final_test_error_list,
                                              filename='classification_layer.csv', loss=final_loss_error_list)
            # save the trained class_net
            torch.save(class_net.state_dict(), BASE_PATH + prefix + 'class_model_state_dict.pt')

    if trial is not None:
        return final_test_error_epoch
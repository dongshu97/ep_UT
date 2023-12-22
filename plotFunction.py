'''
File to plot train_error and test_error over epochs
@author: Jeremie Laydevant
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import json

sns.set_style('white')

def plot_supervised_error(dataframe, idx):
    '''
    Plot train and test error over epochs
    '''

    train_error_tab = dataframe['Train_Error'].to_numpy()
    test_error_tab = dataframe['Test_Error'].to_numpy()

    plt.plot(train_error_tab, label = 'train error #' + str(idx))
    plt.plot(test_error_tab, label = 'test error #' + str(idx))

    plt.ylabel('Error ')
    plt.xlabel('Epochs')

    plt.title('Train and Test error')
    plt.legend()

    return train_error_tab, test_error_tab

def plot_unsupervised_error(dataframe, idx):
    av_error_tab = dataframe['Test_Error_av'].to_numpy()
    max_error_tab = dataframe['Test_Error_max'].to_numpy()

    plt.plot(av_error_tab, label='one2one average error #' + str(idx))
    plt.plot(max_error_tab, label='one2one max error #' + str(idx))

    plt.ylabel('Test Error')
    plt.xlabel('Epochs')

    plt.title('Unspervised test error with label association')
    plt.legend()

    return av_error_tab, max_error_tab


def plot_mean(store_train_error, store_test_error):
    '''
    Plot mean train & test error with +/- std
    '''
    try:
        store_train_error, store_test_error = np.array(store_train_error), np.array(store_test_error)
        mean_train, mean_test = np.mean(store_train_error, axis = 0), np.mean(store_test_error, axis = 0)
        std_train, std_test = np.std(store_train_error, axis = 0), np.std(store_test_error, axis=0)
        epochs = np.arange(0, len(store_test_error[0]))
        plt.figure()
        plt.plot(epochs, mean_train, label = 'mean_train_error')
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, facecolor = '#b9f3f3')

        plt.plot(epochs, mean_test, label = 'mean_test_error')
        plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, facecolor = '#fadcb3')

        plt.ylabel('Error ')
        plt.xlabel('Epochs')
        plt.title('Mean train and Test error with std')
        plt.legend()

    except:
        pass

    return 0

def return_legend(json_path):
    with open(json_path) as f:
        jparams = json.load(f)

    if jparams['batch'] == 1:
        mode = 'sequential mode'
    else:
        mode = 'batch mode'
    structure = str(jparams['fcLayers'][-1])

    for i in range(1, len(jparams['fcLayers'])):
        structure += '-' + str(jparams['fcLayers'][-i-1])

    return structure + '' + mode


def plot_unsupervised_all(store_av_error, unsupervised_legend_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i in len(store_av_error):
        ax.plot(store_av_error[i], label=f'{unsupervised_legend_name[i]}')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Test Error')
    ax.legend()
    ax.titile('Unspervised test error with label association')
    plt.tight_layout()
    plt.savefig('unsupervised_compare_methods.eps', format='eps', dpi=500)

if __name__ == '__main__':
    path = "\\\\?\\"+os.getcwd()
    files = os.listdir(path)
    store_train_error, store_test_error = [], []
    store_av_error, store_max_error = [], []
    supervised_legend_name, unsupervised_legend_name = [], []

    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if not extension == '.py':
            DATAFRAME = pd.read_csv(path + '\\' + simu + '\\results.csv', sep = ',', index_col = 0)
            column_name = DATAFRAME.columns
            if column_name[0] == 'Train_Error':
                train_error_tab, test_error_tab = plot_supervised_error(DATAFRAME, idx)
                legend_smi = return_legend(path + '\\' + simu + '\\config.json')
                supervised_legend_name.append('Supervised'+''+legend_smi)
                store_train_error.append(train_error_tab)
                store_test_error.append(test_error_tab)
            elif column_name[0] == 'Test_Error_av':
                av_error_tab, max_error_tab = plot_unsupervised_error(DATAFRAME, idx)
                legend_smi = return_legend(path + '\\' + simu + '\\config.json')
                unsupervised_legend_name.append('Unsupervised'+''+legend_smi)
        else:
            pass
    if len(store_train_error)!=0 and len(store_test_error)!=0:
        plot_mean(store_train_error, store_test_error)
    elif len(store_av_error)!=0:
        plot_unsupervised_all()



    plt.show()






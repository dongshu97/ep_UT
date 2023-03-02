'''
File to plot train_error and test_error over epochs
@author: Jeremie Laydevant
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_error(dataframe, idx):
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

def plot_mean(store_train_error, store_test_error):
    '''
    Plot mean train & test error with +/- std
    '''
    try:
        store_train_error, store_test_error = np.array(store_train_error), np.array(store_test_error)
        mean_train, mean_test = np.mean(store_train_error, axis = 0), np.mean(store_test_error, axis = 0)
        std_train, std_test = np.std(store_train_error, axis = 0), np.std(store_test_error, axis = 0)
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

if __name__ == '__main__':
    path = "\\\\?\\"+os.getcwd()
    files = os.listdir(path)
    store_train_error, store_test_error = [], []

    for idx, simu in enumerate(files):
        name, extension = os.path.splitext(simu)
        if not extension == '.py':
            DATAFRAME = pd.read_csv(path + '\\' + simu + '\\results.csv', sep = ',', index_col = 0)
            train_error_tab, test_error_tab = plot_error(DATAFRAME, idx)
            store_train_error.append(train_error_tab)
            store_test_error.append(test_error_tab)
        else:
            pass
    plot_mean(store_train_error, store_test_error)


    plt.show()






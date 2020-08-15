from sklearn.datasets import fetch_openml, load_svmlight_file, load_files
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from Encoding.gaussian_fields import GRF_Set
from Encoding.oned_coding import oneD_Encoding_set
from PrintFiles.Imprimir_test import impimir_1D_Encoding, guardar_liquid, impimirFtinessMulti, imprimirBestWeight
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def preprocessing_labels(y):
    '''
    le = preprocessing.LabelEncoder()
    le.fit(y)

    y = le.transform(y)
    return y
    '''
    return preprocessing.LabelEncoder().fit(y).transform(y)

def transformar_dataset(X, encoder):
    '''
    spike_Trains = list()
    for i in range(len(X)):
        spike_Trains.append(encoder.encoding(X[i]))
    return spike_Trains
    '''
    return [encoder.encoding(X[i]) for i in range(len(X))]

def dataset(name):
    if name == 'glass' or name == 'blood' or name == 'balance' or name == 'ionosphere':
        X, y = fetch_openml(name=name, return_X_y=True)  # Glass
        if name == 'ionosphere':
            X = np.delete(X, 1, 1)  # delete second column of X  -> numpy.delete(arr, obj, axis=None) -> arr: input array, obj: numero de column/row, axis: column axis=1, row axis=0
        y = preprocessing_labels(y)

    elif name == 'liver' or name == 'parkinson' or name == 'diabetes' or name == 'breast' or name == 'card' or name == 'fertility':
        df = pd.read_csv(r'C:\Users\carlos\PycharmProjects\gitLSM\SpikesDatasets\\'+name+".csv", header=None)
        X = df.iloc[:, 0:-1].to_numpy()
        y = preprocessing_labels(df.iloc[:, -1].to_numpy())

    elif name == 'iris':
        X, y = datasets.load_iris(return_X_y=True)  # Iris Plant

    elif name == 'wine':
        X, y = datasets.load_wine(return_X_y=True)  # Wine

    return X, y


if __name__ == "__main__":

    experimentos = 40

    X, y = dataset('fertility')  # glass, blood, balance, ionosphere, liver, parkinson, iris, diabetes, wine, breast, card, fertility

    ## Parametros GRF
    # m, simTime, gamma = 4, 10, 1
    ## Parametros 1D encoding
    a, b = 0.01, 9

    for e in range(experimentos):
        # Dividir dataset en train y test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)

        # Transformar dataset de train a Spike Trains
        # encoder = GRF_Set(X_train, simTime, m, gamma, supress_late_spikes=False)
        encoder = oneD_Encoding_set(X_train, a, b)

        spike_trains_Train = transformar_dataset(X_train, encoder)
        spike_trains_Test = transformar_dataset(X_test, encoder)

        impimir_1D_Encoding(spike_trains_Train, y_train, "Fertility_Train"+str(e+1), "labels_train"+str(e+1)) #Agrego
        impimir_1D_Encoding(spike_trains_Test, y_test, "Fertility_Test"+str(e+1), "labels_test"+str(e+1)) #Agrego
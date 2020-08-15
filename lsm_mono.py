from mpi4py import MPI
import numpy as np

from LSM.liquidos.LiquidLambda import liquid_lambda
from LSM.liquidos.LiquidFeedForward import liquid_feedforward
from LSM.SpikesDatasets.cargarDataset import LoadDataset
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from LSM.PrintFiles.Imprimir import txt_stateVector
import sympy

from EvaluateLiquid import EvaluateLiquid
from Item import Item
from Args import arguments

import logging
from itertools import count
import copy
import os


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def evaluarLiquido_Optim(liquido, solution, timeSimulate, spike_Trains, labels, grf=True):
    # Actualizar pesos y delays del liquid
    solucion = np.array([i for i in solution.values])
    liquido.update_W_D(solucion)

    stateVectors = list()
    for p in spike_Trains:
        liquido.reset()
        liquido.injectStimuli(p) if grf else liquido.injectStimuli_1D(p)
        liquido.Simulate(timeSimulate)
        stateVectors.append(liquido.getLiquidState())

    sgd = SGDClassifier()
    acc = Predictions(labels, stateVectors, sgd, True)
    return acc, sgd


def Predictions(labels, stateVectors, sgd, train=True):
    if train:
        sgd.fit(stateVectors, labels)
    return metrics.accuracy_score(labels, sgd.predict(stateVectors))

def simularLiquido(liquido, spike_Trains, timeSimulate, grf=True, predecir=True, sgd=None, Inicial=False, labels=None):
    stateVectors, acc = list(), None

    for p in spike_Trains:
        liquido.reset()
        liquido.injectStimuli(p) if grf else liquido.injectStimuli_1D(p)
        liquido.Simulate(timeSimulate)
        stateVectors.append(liquido.getLiquidState())
    if predecir and Inicial:
        acc = Predictions(labels, stateVectors, sgd)
    elif predecir and Inicial == False:
        acc = Predictions(labels, stateVectors, sgd, False)
    return stateVectors, acc

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(f'lsm_mono_{rank}.log')
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(processName)-10s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.WARNING)

    '''
    Parameters
    '''
    args = arguments()
    tamLiquid = 50
    timeSimulate = 15
    max_evaluations = 10000  # 10000 #100000 #10000
    population_size = 100  # 50#30
    get_bests = 5
    liquid_shape = (5, 5, 2)
    dataset_name = args["dataset"]
    k = args["k"]

    path = os.path.join(os.path.abspath(os.getcwd()), "LSM", "SpikesDatasets")
    results_path = os.path.join(os.path.abspath(os.getcwd()), "LSM", "Results")
    if not os.path.exists(results_path):
        os.makedirs(results_path)


    '''
    Experimento
    '''
    dataset = LoadDataset(path, dataset_name, k)
    if rank==0:
        spikes_train, spikes_test, y_train, y_test = dataset.getSpikesTrain(), dataset.getSpikesTest(), dataset.getLabelsTrain(), dataset.getLabelsTest()
        total_Classes = len(set(y_train))

        # Neuronas de entrada
        inpSize = len(spikes_train[0])  # 1D Coding

        ## Liquido Lambda
        probabilidades = {"Cee": 0.3, "Cei": 0.2, "Cie": 0.4, "Cii": 0.1}
        lammbda = 2
        porc_inh_neurons = 0.2

        # Crear liquido
        l = liquid_feedforward(tamLiquid, inpSize)
        # l = liquid_lambda(tamLiquid, liquid_shape, inpSize, lammbda, porc_inh_neurons, probabilidades)

        # Parametros para conectar estimulo externo al liquido
        conProb, low_w, high_w = 0.2, 0.00001, 1.0

        # Conectar el estimulo externo a las neurona de entrada
        l.connect_ext_stimuli_1D(len(spikes_train[0]), conProb, low_w, high_w)  # 1D Coding

        #### Obtener evaluacion del liquido inicial
        sgd = SGDClassifier()
        stateVectors_Inicial, acc_ini = simularLiquido(l, spikes_train, timeSimulate, grf=False, sgd=sgd, Inicial=True,
                                                    labels=y_train)  # 1D Coding

        txt_stateVector(stateVectors_Inicial, y_train, os.path.join(results_path, f'Inicial_st_Exp_{k}'))

        # Obtener los pesos y retardos del liquido
        pesos, delays = l.get_all_Synapses()

        total_variables = len(pesos)  # Numero de variables a manejar
        total_objetivos = int(sympy.binomial(total_Classes, 2)) + total_Classes  # nCr <- totaldeClases C 2   + total_clases

        #sol = EvaluateComponent(parameters, rank)
        sol = EvaluateLiquid(l, spikes_train, y_train, timeSimulate)

    l = comm.bcast(l if rank==0 else None, root=0)
    spikes_train = comm.bcast(spikes_train if rank==0 else None, root=0)
    spikes_test = comm.bcast(spikes_test if rank==0 else None, root=0)
    y_train = comm.bcast(y_train if rank==0 else None, root=0) 
    y_test = comm.bcast(y_test if rank==0 else None, root=0)
    sol = comm.bcast(sol if rank==0 else None, root=0)
    total_variables = comm.bcast(total_variables if rank==0 else None, root=0)


    '''
    Differential Evolution Parameters
    '''
    de_population = args["population"]
    de_iterations = args["iterations"]
    de_dimensions = total_variables * 2
    de_lb = 0.0001  # -100,
    de_ub = 10.0  # 100,
    de_F = 0.5  # 0.7,
    de_Cr = 1.0  # 0.9,

    # initial population
    pop = []
    for i in range(rank, de_population, size):
        item = Item()
        item.initialize(de_lb, de_ub, de_dimensions)
        item.evaluate(sol)
        pop.append(item)
    
    pop = comm.gather(pop, root=0)
    if rank==0:
        pop = [value for values in pop for value in values]
        logger.warning(EvaluateLiquid.convert(min(pop)))

    for i in range(de_iterations):
        pop = comm.bcast(pop, root=0)
        new_pop = []
        for ii in range(rank, de_population, size):
            item1, item2, item3 = np.random.choice(pop, size=3)
            values = item1.values + de_F * (item2.values - item3.values)

            j = np.random.randint(0, de_dimensions-2, size=1)
            values = np.array([v if r <= de_Cr or c == j else x for c, r, v, x in zip(count(), np.random.random(size=de_dimensions), values, pop[ii].values)])

            values[values<de_lb] = de_lb
            values[values>de_ub] = de_ub

            item = Item(values=values)
            item.evaluate(sol)
            new_pop.append(item if item < pop[ii] else pop[ii])
        pop = comm.gather(new_pop, root=0)

        if rank==0:
            pop = [value for values in pop for value in values]
            logger.warning(f'{i}/{de_iterations}')
            logger.warning(EvaluateLiquid.convert(min(pop)))

    if rank == 0:
        best_result = min(pop)
        acc_Train, sgd = evaluarLiquido_Optim(l, best_result, timeSimulate, spikes_train, y_train, grf=False)
        stateVectors_Best_Testing, acc = simularLiquido(l, spikes_test, timeSimulate, grf=False, sgd=sgd, labels=y_test)

        logger.warning('ACC_Best_Train_Exp_'+f'{k}: ' + f'{acc_Train}')
        logger.warning('ACC_Test_Exp_'+f'{k}: ' + f'{acc}')


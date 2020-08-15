from mpi4py import MPI
import numpy as np

from LSM.liquidos.LiquidLambda import liquid_lambda
from LSM.liquidos.LiquidFeedForward import liquid_feedforward
from LSM.SpikesDatasets.cargarDataset import LoadDataset
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from LSM.PrintFiles.Imprimir import txt_stateVector #, print_accuracies_file, #rastrer_plot, \
    #imprimir_function_values_to_file, imprimir_variables_to_file

from MED.Component import EvaluateLiquid
from Tools import arguments
from Metaheuristics import Item

import logging
import sympy
import copy
import timeit

from itertools import count
import copy
import os

# obtenemos 3 elementos indispensables en MPI, el elemento Comun (comm)
# el elemento actual (rank) y el numero de procesos (size)
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



def experimento(k, tamLiquid, get_bests, path, dataset_name, results_path, liquid_shape):
    dataset = LoadDataset(path, dataset_name, k)
    spikes_train, spikes_test, y_train, y_test = dataset.getSpikesTrain(), dataset.getSpikesTest(), dataset.getLabelsTrain(), dataset.getLabelsTest()
    total_Classes = len(set(y_train))

    # Neuronas de entrada
    inpSize = len(spikes_train[0])  # 1D Coding
    timeSimulate = 15

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

    # Aqui debe de ir el proceso evolutivo
    # Definir parametros del ED
    arg = arguments()
    parameters = {"Population": 5,
                  "Function_Calls": 20,
                  "Dimension": total_variables*2,
                  "LB": 0.0001,  # -100,
                  "UB": 10.0,  # 100,
                  "F": 0.5,  # 0.7,
                  "Cr": 1.0,  # 0.9,
                  "Restart": 5,
                  "Elitism": 5}

    #sol = EvaluateComponent(parameters, rank)
    sol = EvaluateLiquid(l, spikes_train, y_train, timeSimulate, rank)

    # initial population
    item = Item()
    item.initialize(parameters["LB"], parameters["UB"], parameters["Dimension"])
    item.evaluate(sol)

    # item = Item()
    # item.initialize(parameters["LB"], parameters["UB"], parameters["Dimension"])
    # print(timeit.timeit(f'{item.evaluate(sol)}'))
    # return

    pop = comm.gather(item, root=0)

    if rank==0:
        all_pop = [copy.deepcopy(min(pop))]
        logger.warning(EvaluateLiquid.convert(min(pop)))

    function_calls = int(parameters["Function_Calls"]/parameters["Population"])
    for i in range(function_calls):
        pop = comm.bcast(pop, root=0)
        # item1, item2, item3 = np.random.choice(pop, size=3, replace=False)
        item1, item2, item3 = np.random.choice(pop, size=3)
        values = item1.values + parameters['F'] * (item2.values - item3.values)
        # values = pop[r1][0].values + parameters['F'] * (pop[r2][0].values - pop[r3][0].values)

        j = np.random.randint(0, parameters["Dimension"], size=1)
        values = np.array([v if r <= parameters["Cr"] or i == j else x for i, r, v, x in zip(count(), np.random.random(size=parameters["Dimension"]), values, pop[rank].values)])
        #values = np.array([v if r <= parameters["Cr"] or i == j else x for i, r, v, x in zip(count(), np.random.random(size=parameters["Dimension"]), values, pop[i][0].values)])
        #print(values)

        item = Item(values=values)
        item.evaluate(sol)
        pop = comm.gather(item if item < pop[rank] else pop[rank], root=0)

        if rank==0:
            all_pop.append(copy.deepcopy(min(pop)))
            logger.warning(f'{i}/{function_calls}')
            logger.warning(EvaluateLiquid.convert(min(pop)))



    if rank == 0:
        # results = set(all_pop)
        # print(results)
        results = all_pop
        #print(results)

        if "output" in arg:
            printed = []
            with open(arg["output"], "w") as fp:
                for cont, result in enumerate(sorted(results)):
                    if f'{result.fitness:.5f}' in printed:
                        continue

                    fp.write("---------------------------------\n")
                    logger.warning(EvaluateLiquid.convert(result))
                    fp.write(EvaluateLiquid.convert(result))
                    fp.write("\n")

                    printed.append(result.fitness)


                    if cont >= parameters["NumberOfConfigurations"]:
                        break

    if rank == 0:
        best_result = min(all_pop)
        acc_Train, sgd = evaluarLiquido_Optim(l, best_result, timeSimulate, spikes_train, y_train, grf=False)
        stateVectors_Best_Testing, acc = simularLiquido(l, spikes_test, timeSimulate, grf=False, sgd=sgd, labels=y_test)

        logger.warning('ACC_Best_Train_Exp_'+f'{k}: ' + f'{acc_Train}')
        logger.warning('ACC_Test_Exp_'+f'{k}: ' + f'{acc}')




if __name__ == "__main__":
    # set the logger
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler("lsm_mono.log")
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(processName)-10s %(levelname)-8s %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.WARNING)

    tamLiquid = 50
    max_evaluations = 10000  # 10000 #100000 #10000
    population_size = 100  # 50#30
    get_bests = 5
    liquid_shape = (5, 5, 2)

    path_database = os.path.join(os.path.abspath(os.getcwd()), "LSM\SpikesDatasets")
    path_results = os.path.join(os.path.abspath(os.getcwd()), "LSM\Results")
    if not os.path.exists(path_results):
        os.makedirs(path_results)

    # exps = 2
    # for i in range(1, exps+1):
    experimento(1, tamLiquid, get_bests, path_database, "Iris", path_results, liquid_shape)
    #xperimento(1, tamLiquid, get_bests, path_database, "Ionosphere", path_results, liquid_shape)

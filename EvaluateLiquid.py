import numpy as np
import subprocess as sp
import time, os
import logging as log

class EvaluateLiquid:
    """
    Norton Fitness Function
    """
    def __init__(self, liquido, spikeTrains, labels, simTime):
        #self.parameters = parameters
        self.reservoir = liquido
        self.spike_Trains = spikeTrains
        self.simTime = simTime
        self.labels = labels

    def eval(self, solution):
        variabs = np.array([i for i in solution])
        self.reservoir.update_W_D(variabs)

        # obtener el vector de estado de la solucion dentro del liquido
        stateVectors = list()
        for st in self.spike_Trains:
            self.reservoir.reset()
            self.reservoir.injectStimuli_1D(st)
            self.reservoir.Simulate(self.simTime)
            stateVectors.append(self.reservoir.getLiquidState())

        return self.fitness(stateVectors)

    def fitness(self, stateVectors):
        # Hacer el vector de estado en un diccionario para manejarlo mejor acorde a la etqueta de cada clase
        st = dict()
        for key, val in zip(self.labels, stateVectors):
            if key not in st:
                st[key] = list()
            st[key].append(val)

        mu, p = list(), list()

        for keyClass in st:
            classVectors = st[keyClass]
            # M(Ol(t))
            m = np.sum(classVectors, axis=0, out=None) / len(classVectors)
            # p(Ol(t))
            rho = np.sum(np.linalg.norm(m - classVectors, axis=1) / len(classVectors))

            mu.append(m)
            p.append(rho)

        Cd = [np.linalg.norm(mu[i] - mu[j]) for i in range(len(mu)) for j in range(i + 1, len(mu))]
        Cd = sum(Cd) / len(Cd)

        # Calcular Cv
        #Cv = np.average(p)
        Cv = sum(p) / len(p)  #Mas rapido

        # Calcular Sep psi
        Sep = Cd / (Cv + 1)

        return Sep

    def convert(Item):
        try:
            strOut = "Fitness {}\n".format(Item.fitness)
            # for x in range(1, len(Item.values)):
            #     strOut += "{}\n".format(Item.values[x])
            return strOut
        except Exception as Err:
            log.critical(Err)
            return ""
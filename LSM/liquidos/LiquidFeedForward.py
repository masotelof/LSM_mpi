import numpy as np
from LSM.neurons.neuron import SpikingNeuron, SpikeResponseModel
from LSM.PSP.psp import psp
from LSM.Synapses.synapses import StaticSynapse


class liquid_feedforward:

    def __init__(self, reservoirSize, inpSize):
        self.inpSize = inpSize
        self.reservoirSize = reservoirSize
        '''
        self.reservoir = dict()
        # Agregar neuronas GRF
        for i in range(self.inpSize):
            self.reservoir[i] = SpikingNeuron()
        '''
        self.reservoir = {i:SpikingNeuron() for i in range(self.inpSize)}

        # Indice de nueronas aleatorios
        self.rnd_list = np.random.permutation(self.reservoirSize) + self.inpSize

        '''
        # Agregar neuronas al Liquido todas son de la capa oculta
        for i in range(self.inpSize, (self.reservoirSize + self.inpSize)):
            self.reservoir[i] = SpikeResponseModel(psp())
        '''
        self.reservoir.update({i:SpikeResponseModel(psp()) for i in range(self.inpSize, (self.reservoirSize + self.inpSize))})

        # Realizar sinpasis de las neuronas de la capa oculta desde las de la capa de entrada
        #for keyj in self.reservoir.keys():
        for keyj in self.reservoir:
            if isinstance(self.reservoir[keyj], SpikeResponseModel):
                # Agregamos un diccionario vacio a las neuronas SRM para poder llenarlo mas adelante con las
                # conexiones que se usaran de la capa de entrada a la capa "oculta"
                synapses = dict()
                self.reservoir[keyj].setSynapses(synapses)

        # Cada neurona conoce el resto de neuronas en el reservorio para saber su rol
        for i in range(self.inpSize, (self.reservoirSize + self.inpSize)):
            self.reservoir[i].setNeighbors(self.reservoir)

    def connect_ext_stimuli(self, m, caracteristicas, conProb, low_w, high_w):
        for i in range(self.inpSize // m):
            for j in range(self.inpSize, (self.reservoirSize + self.inpSize)):
                # if np.random.uniform() <= conProb:
                for k in range(caracteristicas):
                    self.reservoir[j].getSynapses()[(i * m) + k] = StaticSynapse(np.random.uniform(low_w, high_w), 1)

    def connect_ext_stimuli_1D(self, caracteristicas, conProb, low_w, high_w):
        for j in range(self.inpSize, (self.reservoirSize+self.inpSize)):
            #if np.random.uniform() <= conProb:
            for k in range(caracteristicas):
                self.reservoir[j].getSynapses()[k] = StaticSynapse(np.random.uniform(low_w, high_w), 1)

    def injectStimuli(self, stimuli):
        cont = 0
        for i in stimuli:
            for j in i:
                self.reservoir[cont].setSpike(j[0])
                cont += 1

    def injectStimuli_1D(self, stimuli):
        cont = 0
        for i in stimuli:
            self.reservoir[cont].setSpike(i)
            cont += 1

    def reset(self):
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            self.reservoir[key].setSpike(None)
            if isinstance(self.reservoir[key], SpikeResponseModel):
                ''' Reseatear spikeTrain y v de cada neurona '''
                self.reservoir[key].Voltage_empty()

    def Simulate(self, simTime, startTime=0):
        for t in range(startTime, simTime):
            #for idx in self.reservoir.keys():
            for idx in self.reservoir:
                if isinstance(self.reservoir[idx], SpikeResponseModel):
                    self.reservoir[idx].simulate(t)

    def getLiquidState(self):
        '''
        ls = list()
        for key in self.reservoir.keys():
            if isinstance(self.reservoir[key], SpikeResponseModel):
                ls.append(-1 if self.reservoir[key].getSpike() is None else self.reservoir[key].getSpike())
        return ls
        '''
        return [-1 if self.reservoir[key].getSpike() is None else self.reservoir[key].getSpike() for key in self.reservoir if isinstance(self.reservoir[key], SpikeResponseModel)]

    def get_all_Synapses(self):
        pesos, delays = list(), list()
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            if isinstance(self.reservoir[key], SpikeResponseModel):
                #for idx in self.reservoir[key].getSynapses().keys():
                for idx in self.reservoir[key].getSynapses():
                    w, d = self.reservoir[key].getSynapses()[idx].getSynapse()
                    pesos.append(w)
                    delays.append(d)
        return pesos, delays

    def update_onlyW(self, weights):
        cont = 0
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            if isinstance(self.reservoir[key], SpikeResponseModel):
                #for idx in self.reservoir[key].getSynapses().keys():
                for idx in self.reservoir[key].getSynapses():
                    self.reservoir[key].getSynapses()[idx].setSynapse(weights[cont], 1)
                    cont += 1

    def update_W_D(self, variables):
        weights = variables[:int(len(variables) / 2)]
        delays = variables[int(len(variables) / 2):len(variables)]
        cont = 0
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            if isinstance(self.reservoir[key], SpikeResponseModel):
                for idx in self.reservoir[key].getSynapses():
                    self.reservoir[key].getSynapses()[idx].setSynapse(weights[cont], delays[cont])
                    cont += 1
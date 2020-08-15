import numpy as np
from neurons.neuron import SpikingNeuron, SpikeResponseModel
from PSP.psp import psp
from Synapses.synapses import StaticSynapse


class liquidSRM:

    def __init__(self, reservoirSize, porc_inh_neurons=0.2, inpSize=0, **kwargs):
        self.inpSize = inpSize
        self.reservoirSize = reservoirSize
        # self.inh_neurons = int(self.reservoirSize*inh_neurons)
        # self.exc_neurons = self.reservoirSize-self.inh_neurons
        self.reservoir = dict()
        self.ii = kwargs['ii'] if 'ii' in kwargs else 0.4
        self.ei = kwargs['ei'] if 'ei' in kwargs else 0.2
        self.ie = kwargs['ie'] if 'ie' in kwargs else 0.1
        self.ee = kwargs['ee'] if 'ee' in kwargs else 0.3

        # Agregar neuronas de entrada
        for i in range(self.inpSize):
            self.reservoir[i] = SpikingNeuron()

        # Indice de nueronas aleatorios
        self.rnd_list = np.random.permutation(self.reservoirSize) + self.inpSize

        # Crear una lista de indices aleatorio que representen las neuronas inhi en el reservorio
        self.idx_inh_neurons = [self.rnd_list[i] for i in range(len(self.rnd_list)) if np.random.uniform() < porc_inh_neurons]

        # Agregar neuronas SRM al reservorio
        for i in range(self.inpSize, (self.reservoirSize + self.inpSize)):
            # Asignar neuronas inh al reservorio
            if i in self.idx_inh_neurons:
                self.reservoir[i] = SpikeResponseModel(psp(), neuType='inh')
            else:
                self.reservoir[i] = SpikeResponseModel(psp())

        # Asignar las sinapses a las neuros posinapticas desde las presinpaticas
        for i in range(self.inpSize, (self.reservoirSize + self.inpSize)):
            posinaptic = self.reservoir[i].getType()
            synapses = dict()

            if posinaptic == 1:
                for j in range(self.inpSize, (self.reservoirSize + self.inpSize)):
                    presinaptic = self.reservoir[j].getType()

                    if presinaptic == 1 and np.random.uniform() <= self.ee:
                        synapses[j] = StaticSynapse(np.random.uniform(0.00001, 0.5), 1)

                    elif presinaptic == -1 and np.random.uniform() <= self.ie:
                        synapses[j] = StaticSynapse(np.random.uniform(0.00001, 0.5), 1)

            else:
                for j in range(self.inpSize, (self.reservoirSize + self.inpSize)):
                    presinaptic = self.reservoir[j].getType()

                    if presinaptic == 1 and np.random.uniform() <= self.ei:
                        synapses[j] = StaticSynapse(np.random.uniform(0.00001, 0.5), 1)

                    elif presinaptic == -1 and np.random.uniform() <= self.ii:
                        synapses[j] = StaticSynapse(np.random.uniform(0.00001, 0.5), 1)

            # if not synapses: # ver si no hay synapsis
            self.reservoir[i].setSynapses(synapses)

        # CAda neurona conoce el resto de neuronas en el reservorio
        for i in range(self.inpSize, (self.reservoirSize + self.inpSize)):
            self.reservoir[i].setNeighbors(self.reservoir)

    def connect_ext_stimuli(self, m, caracteristicas, conProb, low_w, high_w):
        for i in range(self.inpSize//m):
            for j in range(self.inpSize, (self.reservoirSize+self.inpSize)):
                if self.reservoir[j].getType() == 1 and np.random.uniform() <= conProb:
                    for k in range(caracteristicas):
                        self.reservoir[j].getSynapses()[(i*m)+k] = StaticSynapse(np.random.uniform(low_w, high_w), 1)


    def connect_ext_stimuli_1D(self, caracteristicas, conProb, low_w, high_w):
        for j in range(self.inpSize, (self.reservoirSize + self.inpSize)):
            for k in range(caracteristicas):
                if self.reservoir[j].getType() == 1 and np.random.uniform() <= conProb:
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
                #     self.reservoir[key].SpikeTrain_empty()
                self.reservoir[key].Voltage_empty()

    def Simulate(self, simTime, startTime=0):
        for t in range(startTime, simTime):
            #for idx in self.reservoir.keys():
            for idx in self.reservoir:
                if isinstance(self.reservoir[idx], SpikeResponseModel):
                    self.reservoir[idx].simulate(t)

    def getLiquidState(self):
        ls = list()
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            if isinstance(self.reservoir[key], SpikeResponseModel):
                ls.append(-1 if self.reservoir[key].getSpike() == None else self.reservoir[key].getSpike())
        return ls

    def get_all_Synapses(self):
        # No se toma las sinapsis de las neuroas de entrada a las neuronas del liquido
        # Solamente las que estan dentro del liquido como tal, (puras SpikeResponseModel)
        pesos, delays = list(), list()
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            if isinstance(self.reservoir[key], SpikeResponseModel):
                #for idx in self.reservoir[key].getSynapses().keys():
                for idx in self.reservoir[key].getSynapses():
                    if isinstance(self.reservoir[idx], SpikeResponseModel):
                        w, d = self.reservoir[key].getSynapses()[idx].getSynapse()
                        pesos.append(w)
                        delays.append(d)
        return pesos, delays

    def update_onlyW(self, weights):
        cont = 0
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            if isinstance(self.reservoir[key], SpikeResponseModel):
                # for idx in self.reservoir[key].getSynapses().keys():
                for idx in self.reservoir[key].getSynapses():
                    if isinstance(self.reservoir[idx], SpikeResponseModel):
                        self.reservoir[key].getSynapses()[idx].setSynapse(weights[cont], 1)
                        cont += 1

    def update_W_D(self, variables):
        weights = variables[:int(len(variables) / 2)]
        delays = variables[int(len(variables) / 2):len(variables)]
        cont = 0
        # for key in self.reservoir.keys():
        for key in self.reservoir:
            if isinstance(self.reservoir[key], SpikeResponseModel):
                #for idx in self.reservoir[key].getSynapses().keys():
                for idx in self.reservoir[key].getSynapses():
                    if isinstance(self.reservoir[idx], SpikeResponseModel):
                        self.reservoir[key].getSynapses()[idx].setSynapse(weights[cont], delays[cont])
                        cont += 1



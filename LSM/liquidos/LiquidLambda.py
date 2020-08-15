import numpy as np
from LSM.neurons.neuron import SpikeResponseModel, SpikingNeuron
from LSM.Synapses.synapses import StaticSynapse
from LSM.PSP.psp import psp


class liquid_lambda:

    def __init__(self, reservoirSize, liquid_shape, inpSize, lammbda, porc_inh_neurons, conn_params):
        self.inpSize = inpSize
        self.reservoirSize = reservoirSize
        self.liquid_shape = liquid_shape
        self.lammbda = lammbda
        self.reservoir = dict()
        self.Cee = conn_params['Cee'] if 'Cee' in conn_params else 0.3
        self.Cei = conn_params['Cei'] if 'Cei' in conn_params else 0.2
        self.Cie = conn_params['Cie'] if 'Cie' in conn_params else 0.4
        self.Cii = conn_params['Cii'] if 'Cii' in conn_params else 0.1

        self.list_positions_3D = self.position_3D(self.liquid_shape)

        # Agregar neuronas de entrada
        for i in range(inpSize):
            self.reservoir[i] = SpikingNeuron()

        # Indice de neuronas aleatorias que formaran el liquido
        self.rnd_list = np.random.permutation(self.reservoirSize) + self.inpSize

        # Indice de nueronas inhibitorias aleatorias a partir de indices aleatorios
        self.idx_inh_list = [i for i in self.rnd_list if np.random.uniform() <= porc_inh_neurons]

        # Agregar neuronas SRM en el reservoir con su posicion
        self.create_liquid()

        # Crear la sinapsis en el liquido en base al modelo lambda
        self.lambda_connection()

        # Cada neurona conoce el resto de neuronas en el reservorio
        for i in range(self.inpSize, (self.reservoirSize + self.inpSize)):
            self.reservoir[i].setNeighbors(self.reservoir)

    def position_3D(self, liquid_shape):
        # Centrar el eje y
        pop_y = 0

        # centrar el eje x en cero
        pop_x = liquid_shape[0] / 2.0

        # centrar el eje z en cero
        pop_z = liquid_shape[2] / 2.0

        # Generar las posicion espaciales a partir de la tupla (x,y,z)
        lista_pos = list()
        for i in range(liquid_shape[0]):
            for j in range(liquid_shape[1]):
                for k in range(liquid_shape[2]):
                    lista_pos.append((i + pop_x, j + pop_y, k + pop_z))

        return lista_pos

    def create_liquid(self):
        pos = 0
        for i in self.rnd_list:
            if i in self.idx_inh_list:
                self.reservoir[i] = SpikeResponseModel(psp(), neuType='inh', position=self.list_positions_3D[pos])
            else:
                self.reservoir[i] = SpikeResponseModel(psp(), position=self.list_positions_3D[pos])
            pos += 1

    def euclideanDistance(self, pointA, pointB):
        """Calcular la Distancia Euclidea entre dos puntos 3D
            pointA = tuple(x1,y1,z1)
           pointB = tuple(x2,y2,z2)
        """
        x1, y1, z1 = pointA
        x2, y2, z2 = pointB
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def lambda_connection(self):
        # Calcular la probabilidad de conexion PI
        for j in range(self.inpSize, self.reservoirSize + self.inpSize):  # Posinaptic neuron
            synapses = dict()
            for i in range(self.inpSize, self.reservoirSize + self.inpSize):  # Presinaptic neuron
                # flag = False
                D = self.euclideanDistance(self.reservoir[j].getPosition(), self.reservoir[i].getPosition())
                if self.reservoir[i].getType() == 1 and self.reservoir[j].getType() == 1:  # SEE (pre-post) i -> j
                    Pij = self.Cee * np.exp(-(D / self.lammbda) ** 2)
                elif self.reservoir[i].getType() == 1 and self.reservoir[j].getType() == 0:  # SEI (pre-post) i -> j
                    Pij = self.Cei * np.exp(-(D / self.lammbda) ** 2)
                elif self.reservoir[i].getType() == 0 and self.reservoir[j].getType() == 1:  # SIE (pre-post) i -> j
                    Pij = self.Cie * np.exp(-(D / self.lammbda) ** 2)
                    # flag = True
                else:  # SII (pre-post) i -> j
                    Pij = self.Cii * np.exp(-(D / self.lammbda) ** 2)
                    # flag = True

                # si Pij es mayor que un numero aleatorio, entonces se hace esa conexion
                # if self.umbral_conx <= Pij:
                if np.random.uniform() < Pij:
                    synapses[i] = StaticSynapse(np.random.uniform(0.00001, 0.5), 1)

            self.reservoir[j].setSynapses(synapses)

    def connect_ext_stimuli(self, m, caracteristicas, conProb, low_w, high_w):
        for i in range(self.inpSize // m):
            for j in range(self.inpSize, (self.reservoirSize + self.inpSize)):
                if self.reservoir[j].getType() == 1 and np.random.uniform() <= conProb:
                    for k in range(caracteristicas):
                        self.reservoir[j].getSynapses()[(i * m) + k] = StaticSynapse(np.random.uniform(low_w, high_w),
                                                                                     1)

    def connect_ext_stimuli_1D(self, caracteristicas, conProb, low_w, high_w):
        for j in range(self.inpSize, (self.reservoirSize + self.inpSize)):
            for k in range(caracteristicas):
                if self.reservoir[j].getType() == 1 and np.random.uniform() <= conProb:
                    self.reservoir[j].getSynapses()[k] = StaticSynapse(np.random.uniform(low_w, high_w), 1)

    def reset(self):
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            self.reservoir[key].setSpike(None)
            if isinstance(self.reservoir[key], SpikeResponseModel):
                ''' Reseatear spikeTrain y v de cada neurona '''
                # self.reservoir[key].SpikeTrain_empty()
                self.reservoir[key].Voltage_empty()

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
                # for idx in self.reservoir[key].getSynapses().keys():
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
        #for key in self.reservoir.keys():
        for key in self.reservoir:
            if isinstance(self.reservoir[key], SpikeResponseModel):
                #for idx in self.reservoir[key].getSynapses().keys():
                for idx in self.reservoir[key].getSynapses():
                    if isinstance(self.reservoir[idx], SpikeResponseModel):
                        self.reservoir[key].getSynapses()[idx].setSynapse(weights[cont], delays[cont])
                        cont += 1

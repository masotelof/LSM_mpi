import os
#import matplotlib.pyplot as plt
#from jmetal.util.solution import print_function_values_to_file, print_variables_to_file
import json

def txt_stateVector(stateVectors, labels, filename: str):
    '''
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass
    '''
    st = dict()
    '''
    for i in range(len(labels)):
        class_Index = labels[i]
        if not st.__contains__(class_Index):
            st.__setitem__(labels[i], list())
        st[class_Index].append(stateVectors[i])
    '''
    for key, val in zip(labels, stateVectors):
        if key not in st:
            st[key] = list()
        st[key].append(val)

    with open(filename, 'w') as of:
        of.write(json.dumps(st))
        '''
        #for key in st.keys():
        for key in st:
            of.write("Class: " + str(key))
            of.write('\n')
            for i in st[key]:
                of.write(str(i))
                of.write('\n')
        '''
    #of.close()


# def print_stateVector_sorted(stateVectors, labels):
#     st = dict()
#     '''
#     for i in range(len(labels)):
#         class_Index = labels[i]
#         if not st.__contains__(class_Index):
#             st.__setitem__(labels[i], list())
#         st[class_Index].append(stateVectors[i])
#     '''
#     for key, val in zip(labels, stateVectors):
#         if key not in st:
#             st[key] = list()
#         st[key].append(val)
#     # for key in st.keys():
#
#     for key in st:
#         of.write(json.dumps(st))
#         '''
#         print("Class: " + str(key))
#         for i in st[key]:
#             print(i)
#         '''


def print_accuracies_file(accuracys, filename: str):
    '''
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except FileNotFoundError:
        pass
    '''
    with open(filename, 'w') as of:
        '''
        for i, accuracy in enumerate(accuracys):
            of.write(str(accuracy))
            of.write('\n')
        '''
        of.write("\n".join(str(accuracy) for accuracy in accuracys))
        
    #of.close()


# def rastrer_plot(colors, spikes, numero, key, sizeSpike=0.5):
#     plt.eventplot(spikes, color=colors, linelengths=sizeSpike)
#     plt.title("Clase: " + str(key))
#     plt.xlabel('Spike at time')
#     plt.ylabel('Neuron')
#     plt.show()
#
#
# # Save results to file
# def imprimir_function_values_to_file(solutions, name):
#     #print_function_values_to_file(solutions, 'FUN_Parallel_Exp_' + str(num))
#     print_function_values_to_file(solutions, name)
#
#
# def imprimir_variables_to_file(solutions, name):
#     #print_variables_to_file(solutions, 'VAR_Parallel_Exp_' + str(num))
#     print_variables_to_file(solutions, name)

from data import generateGraph
from perceptron import saveIterations
import numpy as np
import matplotlib.pyplot as plt

# loss function used for the Adaline algorithm
# based on 2nd norm of 2 vectors
def Lss(S,w):
    ans = 0
    for i in range(len(S)):
        ans += (S[i][1] - w.T @ S[i][0])**2
    return ans/len(S)

# the Adaline Algorithm with the delta rule
# the learning rate is fixed on 0.0001
def adaline(S, w, Tmax, hyperplan, viz_mode, visualize = False, save = False):
    '''
    S: the dataset, composed of tuples (xi,yi)
    w: initial weight vector
    Tmax: upper bound of the iterations of the Adaline algorithm 
    hyperplan: the equation of the hyperplan to find
    viz_mode: either "2d" or "3d"
    vizualize: turn True to plot the hyperplan and the data, False by default
    save: turn True to save return also a list of the intermediate PLA results
    '''
    t_list = []
    conv_list = []
    w_list=[]
    for t in range(Tmax):
        for i in range(len(S)):
            if (S[i][1] - w.T @ S[i][0]) != 0 : 
                for j in range(len(w)):
                    w[j][0] += 2 * 0.0001 * (S[i][1] - (w.T @ S[i][0])) * S[i][0][j]
        if save : saveIterations(w_list, t_list, conv_list, w, t, Lss(S,w)[0])
    if visualize : generateGraph(S, w, hyperplan, mode=viz_mode)
    if save : return (w, Lss(S,w)[0], t, t_list, conv_list, w_list)
    else : return (w, Lss(S,w)[0], t)
from perceptron import PLA, Ls, saveIterations
import numpy as np
from data import generateGraph

# the Pocket Algorithm
def Pocket(S, w, Tmax, hyperplan, viz_mode, visualize = False, save = False):
    '''
    S: the dataset, composed of tuples (xi,yi)
    w: initial weight vector
    Tmax: upper bound of the iterations of the pocket algorithm 
    hyperplan: the equation of the hyperplan to find
    viz_mode: either "2d" or "3d"
    vizualize: turn True to plot the hyperplan and the data, False by default
    save: turn True to save return also a list of the intermediate PLA results
    '''
    t_list = []
    conv_list = []
    w_list=[]
    t = 0
    w0 = np.array(w)
    n = len(S)
    conv = Ls(S,w,n)
    while t < Tmax and conv:
        for i in range(n):
            if np.sign(w.T @ S[i][0]) * S[i][1] < 0:
                for j in range(len(w)):
                    w[j][0] += S[i][1] * S[i][0][j]
        t += 1
        conv = Ls(S,w,n)
        if save : saveIterations(w_list, t_list, conv_list, w0, t, conv)
        if Ls(S, w0, len(S)) > Ls(S, w, len(S)) : w0 = w
    if visualize : generateGraph(S, w0, hyperplan, mode = viz_mode)
    if save : return (w0, Ls(S, w0, len(S)), t, t_list, conv_list, w_list)
    else : return (w0, Ls(S, w0, len(S)), t)

from perceptron import PLA, Ls, saveIterations
import numpy as np
from data import generateGraph

def Pocket(S, w, Tmax, hyperplan, viz_mode, visualize = False, save = False): 
    t_list = []
    conv_list = []
    w_list=[]
    t = 0
    conv = 1
    w0 = np.array(w)
    while t < Tmax and conv:
        n = len(S)
        conv = Ls(S,w,n)
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

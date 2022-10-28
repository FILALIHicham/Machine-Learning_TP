from data import generateGraph, hyperplan
from perceptron import saveIterations
import numpy as np
import matplotlib.pyplot as plt

def Lss(S,w):
    ans = 0
    for i in range(len(S)):
        ans += (S[i][1] - w.T @ S[i][0])**2
    return ans/len(S)

def adaline(S, w, Tmax, hyperplan, viz_mode, visualize = False, save = False):
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
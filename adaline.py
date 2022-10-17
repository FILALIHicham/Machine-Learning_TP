from data import generateGraph
from perceptron import hyperplan
import numpy as np
import matplotlib.pyplot as plt

def Lss(S,w):
    ans = 0
    for i in range(len(S)):
        ans += (S[i][1] - w.T @ S[i][0])**2
    return ans/len(S)

def adaline(S, w, Tmax, hyperplan, visualize = False):
    for t in range(Tmax):
        for i in range(len(S)):
            if (S[i][1] - w.T @ S[i][0]) != 0 : 
                for j in range(len(w)):
                    w[j][0] += 2 * 0.0001 * (S[i][1] - (w.T @ S[i][0])) * S[i][0][j]
    if visualize : generateGraph(S, w, hyperplan)
    return (w, Lss(S,w), t+1)
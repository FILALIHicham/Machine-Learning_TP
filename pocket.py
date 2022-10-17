from perceptron import PLA, Ls
import numpy as np
from data import generateGraph

def Pocket(S, w, Tmax, hyperplan, visualize = False): 
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
        if Ls(S, w0, len(S)) > Ls(S, w, len(S)) : w0 = w
    if visualize : generateGraph(S, w0, hyperplan)
    return (w0, Ls(S, w0, len(S)), t+1)
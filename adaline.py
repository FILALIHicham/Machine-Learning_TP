from pocket import generateData
import numpy as np
import matplotlib.pyplot as plt

def Lss(S,w):
    ans = 0
    for i in range(len(S)):
        ans += (S[i][1] - w.T @ S[i][0])**2
    return ans/len(S)

def adaline(S, w, Tmax):
    for t in range(Tmax):
        for i in range(len(S)):
            if (S[i][1] - w.T @ S[i][0]) != 0 : 
                for j in range(len(w)):
                    w[j][0] += 2 * (S[i][1] - (w.T @ S[i][0])) * S[i][0][j]
    return (w,t, Lss(S,w))
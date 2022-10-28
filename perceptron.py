import numpy as np 
from data import generateGraph

# function that returns 0 if X == Y, 1 otherwise
def kronecker(x,y,w):
    y_pred = np.sign(w.T @ x)
    if y != y_pred : return 1
    return 0

# loss function
def Ls(S, w, n):
    ans = 0
    for i in range(n):
        ans += kronecker(S[i][0], S[i][1], w)
    return ans/n

def saveIterations(w_list, t_list, conv_list, w, t, conv):
    w_list.append(w)
    t_list.append(t)
    conv_list.append(conv)

# the Perceptron algorithm
def PLA(S, w0, hyperplan, viz_mode, visualize = False, save = False):
    t_list = []
    conv_list = []
    w_list=[]
    n = len(S)
    t = 0
    w = np.array(w0)
    conv = Ls(S,w,n)
    while conv :
        for i in range(n):
            if np.sign(w.T @ S[i][0]) * S[i][1] < 0:
                for j in range(len(w)):
                    w[j][0] += S[i][1] * S[i][0][j]
        t += 1
        conv = Ls(S,w,n)
        if save : saveIterations(w_list, t_list, conv_list, w0, t, conv)
    if visualize : generateGraph(S, w, hyperplan, mode=viz_mode)
    if save : return (w, conv, t , t_list, conv_list, w_list)
    else : return (w, conv, t)



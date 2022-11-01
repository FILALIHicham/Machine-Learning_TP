import numpy as np 
from data import generateGraph

# function that returns 0 if y == y_pred, 1 otherwise
def kronecker(x,y,w):
    '''
    x: vector e.i. 1 line in the dataset
    y: the list of x (-1 or 1)
    w: the weight vector
    '''
    y_pred = np.sign(w.T @ x)
    if y != y_pred : return 1
    return 0

# loss function based on the Kronecker function
# returns the percentage of misclassified vectors
def Ls(S, w, n):
    '''
    S: the dataset, composed of tuples (xi,yi)
    w: weight vector
    n: size of S
    '''
    ans = 0
    for i in range(n):
        ans += kronecker(S[i][0], S[i][1], w)
    return ans/n

# helper function to save the results of the optimul search at each iteration
def saveIterations(w_list, t_list, conv_list, w, t, conv):
    w_list.append(w)
    t_list.append(t)
    conv_list.append(conv)

# the Perceptron Learning Algorithm
def PLA(S, w0, hyperplan, viz_mode, visualize = False, save = False):
    '''
    S: the dataset, composed of tuples (xi,yi)
    w0: initial weight vector
    hyperplan: the equation of the hyperplan to find
    viz_mode: either "2d" or "3d"
    vizualize: turn True to plot the hyperplan and the data, False by default
    save: turn True to save return also a list of the intermediate PLA results
    '''
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



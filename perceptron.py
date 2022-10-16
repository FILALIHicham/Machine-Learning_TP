import numpy as np 
import matplotlib.pyplot as plt 

# function that returns 1 if X == Y, 0 otherwise
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


def PLA(S, w0, visualize = False):
    n = len(S)
    t = 0
    w = np.array(w0)
    conv = Ls(S,w,n)
    while conv :
        print(conv)
        for i in range(n):
            if np.sign(w.T @ S[i][0]) * S[i][1] < 0:
                for j in range(len(w)):
                    w[j][0] += S[i][1] * S[i][0][j]
        t += 1
        conv = Ls(S,w,n)
    return (w, conv, t)

from random import randint, random, choice

# S = []
# for i in range(100):
#     c = [0,1]
#     if choice(c) :
#         xi = [1,randint(0,10), randint(0,20)]
#         yi = 1
#     else:
#         xi = [1,randint(11,30), randint(22,40)]
#         yi = -1
#     S.append((xi,yi))




# w0 = np.array([[0.76], [0.46], [0.26]])
# wop, conv, t = PLA(S, w0)
# print(wop, conv, t)

def hyperplan(wop,x):
    return -(wop[1][0]*x + wop[0][0])/wop[2][0]

# for e in S:
#     if e[1] == 1 :plt.plot(e[0][1], e[0][2], "o", color = 'red')
#     else :plt.plot(e[0][1], e[0][2], "*", color = 'green')
# x = [0,30]
# y = [hyperplan(wop, x[0]),hyperplan(wop, x[1])]
# plt.plot(x, y,'-b')
# plt.show()
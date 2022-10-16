import numpy as np 
import matplotlib.pyplot as plt 
from random import randint, random, choice

def generateData(s):
    S = []
    for i in range(s):
        c = [0,1]
        if choice(c) :
            xi = [1,randint(0,15), randint(0,25)]
            yi = 1
        else:
            xi = [1,randint(10,30), randint(15,40)]
            yi = -1
        S.append((xi,yi))
    return S

from perceptron import PLA, Ls, hyperplan

def Pocket(S,w,Tmax):
    w = PLA(S,w)[0]
    for t in range(Tmax):
        wtemp = PLA(S,w)[0]
        if Ls(S,w,len(S)) > Ls(S,wtemp,len(S)) : w = wtemp
    return (w, t, Ls(S,w,len(S)))

# S = generateData(100)

# w0 = np.array([[0.76], [0.46], [0.26]])
# wop, conv, t = Pocket(S, w0, 50)
# print(wop, conv, t)

# for e in S:
#     if e[1] == 1 :plt.plot(e[0][1], e[0][2], "o", color = 'red')
#     else :plt.plot(e[0][1], e[0][2], "*", color = 'green')
# x = [0,30]
# y = [hyperplan(wop, x[0]),hyperplan(wop, x[1])]
# plt.plot(x, y,'-b')
# plt.show()


from random import randint, random, choice
import matplotlib.pyplot as plt


def generateGraph(S, wop, hyperplan):
    for e in S:
        if e[1] == 1 :plt.plot(e[0][1], e[0][2], "o", color = 'red')
        else :plt.plot(e[0][1], e[0][2], "*", color = 'green')
    x = [0,20]
    y = [hyperplan(wop, x[0]),hyperplan(wop, x[1])]
    plt.plot(x, y,'-b')
    plt.show()

def generateData(s,x1,x2,x3,x4,y1,y2,y3,y4):
    S = []
    for i in range(s):
        c = [0,1]
        if choice(c) :
            xi = [1,randint(x1,x2), randint(y1,y2)]
            yi = 1
        else:
            xi = [1,randint(x3,x4), randint(y3,y4)]
            yi = -1
        S.append((xi,yi))
    return S
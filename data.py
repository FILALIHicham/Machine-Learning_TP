from random import randint, random, choice
import matplotlib.pyplot as plt
import numpy as np


# hyperplan equation (in 2D)
def hyperplan(wop,x):
    return -(wop[1][0]*x + wop[0][0])/wop[2][0]

# hyperplan equation (in 2D)
def hyperplan3D(wop,x,y):
    return -(wop[1][0] * x + wop[2][0] * y + wop[0][0])/wop[3][0]

def generateGraph(S, wop, hyperplan, mode):
    if mode == "2d":
        for e in S:
            if e[1] == 1 :plt.plot(e[0][1], e[0][2], "o", color = 'red')
            else :plt.plot(e[0][1], e[0][2], "o", color = 'green')
        x = [0,20]
        y = [hyperplan(wop, x[0]),hyperplan(wop, x[1])]
        plt.plot(x, y,'-b')
    elif mode == "3d":
        # Create the figure
        fig = plt.figure()

        # Add an axes
        ax = fig.add_subplot(111,projection='3d')
        for e in S:
            if e[1] == 1 :ax.plot(e[0][1], e[0][2], e[0][3], "o", color = 'red')
            else : ax.plot(e[0][1], e[0][2], e[0][3], "o", color = 'green')
        x, y = np.meshgrid(range(20),range(20))
        z = hyperplan3D(wop, x, y)
        ax.plot_surface(x, y, z)
    plt.show()

def generateData(data_size,x1,x2,x3,x4,y1,y2,y3,y4):
    S = []
    for i in range(data_size):
        c = [0,1]
        if choice(c) :
            xi = [1,randint(x1,x2), randint(y1,y2)]
            yi = 1
        else:
            xi = [1,randint(x3,x4), randint(y3,y4)]
            yi = -1
        S.append((xi,yi))
    return S


def generateData3D(s,x,y,z):
    S = []
    for i in range(s):
        c = [0,1]
        if choice(c) :
            xi = [1,randint(x[0],x[1]), randint(y[0],y[1]), randint(z[0],z[1])]
            yi = 1
        else:
            xi = [1,randint(x[2],x[3]), randint(y[2],y[3]), randint(z[0],z[1])]
            yi = -1
        S.append((xi,yi))
    return S


# import matplotlib.animation as animation

# fig, ax = plt.subplots()

# x = [0,20]
# y = [hyperplan(wop, x[0]),hyperplan(wop, x[1])]
# line, = ax.plot(x, y)

# def calc_y(w_list, x, hyperplan):
#     return [[hyperplan(w, x[0]), hyperplan(w, x[1])] for w in w_list]


# def init():
#     line.set_data([],[])
#     return line

# def animate(i):
#     line.set_data(x,Y[i])  # update the data.
#     return line,



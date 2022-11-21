import numpy as np
# y = a*x + b
def loss(a,b,x,y):
    err = 0
    for i in range(len(x)):
        err += (y[i] - (a * x[i] + b))**2
    return err/float(len(x))

def gradient(a,b,x,y,rate):
    a_grad, b_grad = 0, 0
    n = len(x)

    for i in range(n):
        a_grad += -(2/n) * x[i] * (y[i] - (a * x[i] + b))
        b_grad += -(2/n) * (y[i] - (a * x[i] + b))

    a0 = a - a_grad * rate
    b0 = b - b_grad * rate
    return a0, b0

def linearReg(x,y):
    n = len(x)
    Sx = sum(x)
    Sf = sum(y)
    Sxx = sum([i**2 for i in x])
    Sfx = sum(x[i] * y[i] for i in range(n))
    b = (Sxx * Sf - Sx * Sfx)/(n * Sxx - Sx**2)
    a = (-Sx * Sf + n * Sfx)/(n * Sxx - Sx**2)
    return a, b

def Lossf(x, y):
    Sff = sum([i**2 for i in y])
    n = len(x)
    Sx = sum(x)
    Sf = sum(y)
    Sxx = sum([i**2 for i in x])
    Sfx = sum(x[i] * y[i] for i in range(n))
    Sffpred = ((Sf ** 2) * Sxx - 2 * Sf * Sx * Sfx + n * (Sfx ** 2))/(n * Sxx - (Sx ** 2))
    return (Sff - Sffpred)
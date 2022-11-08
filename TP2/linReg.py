import numpy as np

# y = c + x * m : regression line => 

# def linReg(X,Y):
#     n = len(X)
#     SX, SY = sum(X), sum(Y)
#     m = (n * sum([X[i] * Y[i] for i in range(n)]) - SX * SY)/(n*sum([x**2 for x in X]) - SX ** 2)
#     c = (SY - m * SX)/n
#     return (m, c)

# def loss(y, y_pred):
#     return sum([(y[i] - y_pred[i])**2 for i in range(len(y))])/len(y)


def Ls(X, y, w): # X is a matrix
    return sum([(y[i] - w.dot(X[i]))**2 for i in range(len(y))]) / len(y)

def gradLs(X, y, w):
    return 2.0 * sum([X[i].dot(y[i] - w.dot(X[i])) for i in range(len(y))]) / len(y)

def linRegOpt(X, y, e):
    w = np.zeros(len(X[0]))
    while gradLs(X, y, w) > e:
        w -= 0.001 * gradLs(X, y, w)
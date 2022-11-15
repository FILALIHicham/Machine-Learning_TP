import numpy as np

# sigmpid activation function
def sigmoid(x, w):
    '''
    x: a vector of features
    w: weight vector
    '''
    return 1/(1 + np.exp(- w.T @ x))

# cost function of the logistic regression
def costLogReg(X,Y,h,w):
    """
    X: table of vectors x_i
    Y: list of labels of each vector x_i in X
    h: activation function
    """
    m = len(X)
    som = 0
    for i in range(m):
        som += -Y[i] * np.log(h(X[i],w)) - (1 - Y[i]) * np.log(1 - h(X[i],w))
    return som/m

# gradient of cost function
def gradCost(X,Y,h,w):
    """
    X: table of vectors x_i
    Y: vector of labels y_i
    h: activation function
    """
    grad = []
    d = len(w)
    m = len(X)
    for j in range(d):
        som = 0
        for i in range(m):
            som += (h(X[i],w) - Y[i]) * X[i][j]
        grad.append(som/m)
    return grad

# logistic regression algorithm
def RergessionLogistic(X,Y,h,w):
    compteur = 0 #initialisation de compteur
    learning_rate = 0.01
    gradloss = gradCost(X,Y,h,w)
    while(np.linalg.norm(gradloss) > 0.01):
        #update gradient
        for i in range(len(w)):
            w[i] -= learning_rate *  gradloss[i]
        gradloss = gradCost(X,Y,h,w)
        print(np.linalg.norm(gradloss))
        compteur+=1
    loss = costLogReg(X,Y,h,w)
    print(loss)
    print("le nombre des iterations est : ", compteur)
    return w

import numpy as np

# sigmpid activation function
def sigmoid(x):
    '''
    x: a vector
    '''
    return 1/(1 + np.exp(x))

# cost function of the logistic regression
def costLogReg(x,y,w):
    """
    x: a vector x_i
    y: the label y_i associated
    w: weight vector
    """
    return np.log(1 + np.exp(-y * np.dot(w,x)))

# gradient of cost function
def gradCost(x,y,h,w):
    """
    x: a vector x_i
    y: the label y_i associated
    w: weight vector
    h: activation function
    """
    grad = []
    d = len(w)
    fact = h(- y * np.dot(w,x))
    grad = [-y * np.exp( -y * np.dot(w,x)) * x[i] for i in range(d)]
    return np.dot(fact,grad)

def Grad(X,Y,h,w):
    m = len(X)
    d = len(w)
    L = [0] * d
    for row in range(m):
        x = gradCost(X[row], Y[row], h, w)
        for i in range(m):
            L[i] += x[i]
    for i in range(d):
        L[i] /= m
    return L

# logistic regression algorithm
def RergessionLogistic(X,Y,h,w):
    compteur = 0 #initialisation de compteur
    lr = 0.01
    loss = Grad(X,Y,h,w)
    while(np.linalg.norm(loss) > 0.01):
        #update gradient
        for i in range(len(w)):
            w[i] -= lr *  loss[i]
        loss = Grad(X,Y,h,w)
        print(np.linalg.norm(loss))
        compteur += 1
    loss = costLogReg(X,Y,w)
    print(loss)
    print("le nombre des iterations est : ", compteur)
    return w

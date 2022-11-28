#!/usr/bin/env python
# coding: utf-8

# In[8]:


#----importation de les packages------------------------#
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


# In[10]:


#___importation de data______________#
dataframe = pd.read_csv('binary.csv')
#print(dataframe)
# Définir notre variable dépendante y et nos varaibles indépendantes X
#_____les feautures x1 et x2__________#
x1_x2_x3 = dataframe.iloc[:, [1, 2,3]].values
#print(data)

#_____label____________#
targets =  dataframe.iloc[:, [0]].values
#print(Y)

#---la visualisation de data----------#
plt.title('la visualisation de data ')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1_x2_x3[:, 0], x1_x2_x3[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
plt.show()


# In[ ]:


#---la standarisation de data----#
sc = StandardScaler()
X = sc.fit_transform(x1_x2)
#print(X)


#--ajout 1 pour le bais----#
m = len(X)
one = np.ones((m, 1))
X = np.hstack((one, X))
#print(X)


# In[ ]:


#---la standarisation de data----#
sc = StandardScaler()
X = sc.fit_transform(x1_x2)
#print(X)


#--ajout 1 pour le bais----#
m = len(X)
one = np.ones((m, 1))
X = np.hstack((one, X))
#print(X)


# In[1]:


#----importation de les packages------------------------#
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


#___importation de data______________#
dataframe = pd.read_csv('binary.csv')
#print(dataframe)


# Définir notre variable dépendante y et nos varaibles indépendantes X
#_____les feautures x1 et x2__________#
x1_x2 = dataframe.iloc[:, [1, 2]].values
#print(data)

#_____label____________#
targets =  dataframe.iloc[:, [0]].values
#print(Y)
'''
#---la visualisation de data----------#
plt.title('la visualisation de data ')
plt.xlabel('x1')
plt.ylabel('x2')
plt.scatter(x1_x2[:, 0], x1_x2[:, 1], s=40, c=targets, cmap=plt.cm.Spectral)
plt.show()
'''
#---la standarisation de data----#
sc = StandardScaler()
X = sc.fit_transform(x1_x2)
#print(X)


#--ajout 1 pour le bais----#
m = len(X)
one = np.ones((m, 1))
X = np.hstack((one, X))
#print(X)


#------------la fonction sigmoid---------------#
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


#--------------lossfunction-------------------------------#
def lossfunction(x, y, w):
    som = 0
    for i in range(m):
         som += np.log(1 + np.exp(-y[i]*(np.dot(w, x[i]))))

    return som/m

def gradient(x, y, w):
    som = 0
    for i in range(m):
        t1 = (-y[i] * np.exp(-y[i] * np.dot(w, x[i])))
        t2 = 1/(1 + np.exp(-y[i] * np.dot(w, x[i])))
        som +=  (t1 * t2) * x[i]
    return som/m


#--------Algorithme de regression logistique---------------#
def RergessionLogistic(x, targets, w):
    compteur = 0 #initialisation de compteur
    learning_rate = 2 
    gradloss = gradient(x, targets, w)
    while(np.linalg.norm(gradloss) > 0.001):
        #update gradient
        w =  w - learning_rate *  gradloss
        gradloss = gradient(x, targets, w)
        print(np.linalg.norm(gradloss))
        compteur+=1
    loss = lossfunction(x, targets, w)
    print(loss)
    print("le nombre des iterations est : ", compteur)
    return w


# In[2]:


#initialisation de w
w = [0, 0, 0]
theta = RergessionLogistic(X, targets, w)


#-------la visualisation des resultats dans 2D---------------------#
plt.scatter(X[:, 1], X[:, 2], s=40, c=targets, cmap=plt.cm.Spectral)
print('w* = ', theta)
plt.axis('scaled')
if ( theta[1]!= 0 ):
    t = np.linspace(-3, 2, 2)
    z = (-theta[0]/theta[1])*t - theta[2]/theta[1]
    plt.title('Beste separateur')
    plt.plot(t, z, color='green')
    

#-------la visualisation des resultats dans 3D---------------------#
fig1 = plt.figure()
ax1 = plt.axes(projection = '3d')
xo, zo = np.linspace(min(X[:,1])-1, max(X[:,1])+1, 100), np.linspace(min(X[:,2])-1, max(X[:,2])+1, 1000)
xo,zo = np.meshgrid(xo, zo)
y1 = 1/(1+np.exp(-theta[0]*xo - theta[1]*zo - theta[2]))
ax1.scatter3D(X[:,1],X[:,2],targets,c=targets, cmap=plt.cm.Spectral)
ax1.plot_surface(xo, zo, y1, alpha=0.3)
ax1.set_xlim3d(min(X[:,1])-1,max(X[:,2])+1)
ax1.set_ylim3d(min(X[:,1])-1,max(X[:,2])+1)
ax1.set_zlim3d(0, 2)
plt.show()






# In[ ]:





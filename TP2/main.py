from matplotlib import pyplot as plt
import numpy as np
from linReg import *
import csv

X,Y = [],[]

# open and format data 
with open('car data.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append(int(row["speed"]))
        Y.append(int(row["dist"]))

# calculate the regression hyperplan slop and intercept (2D case)
# m, c = linReg(X,Y)
# y_pred = [x*m + c for x in X]

# # show results
# print("Optimal slope =",m , "| Optimal intercept =",c)
# print("Empirical Error Value =", loss(Y, y_pred))

# # plot results
plt.plot(X,Y,'o')
# plt.plot(X, y_pred,'-b')
plt.show()
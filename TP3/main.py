from matplotlib import pyplot as plt
import numpy as np
from polyReg import *
import csv

X,Y = [],[]

# open and format data 
with open('pressure.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append(int(row["temperature"]))
        Y.append(float(row["pressure"]))

# calculate the regression hyperplan slop and intercept (2D case)
# m, c = linReg(X,Y)
# y_pred = [x*m + c for x in X]

# # show results
# print("Optimal slope =",m , "| Optimal intercept =",c)
# print("Empirical Error Value =", loss(Y, y_pred))

# # plot results
plt.plot(X,Y,'-')
plt.plot(X,Y,'o')
plt.xlabel("temperature")
plt.ylabel("pressure")
plt.title("Temperature per pression graph")
plt.show()
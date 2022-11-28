import csv
from nonLinTr import *

X, Y = [],[]

# open and format data 
with open('TP4/ex2data2.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        X.append([1,float(row["x1"]), float(row["x2"])])
        Y.append(int(row["y"]))

w = np.array([0.0,0.0,0.0])
w0 = RergessionLogistic(X,Y,sigmoid,w)
print(w0)
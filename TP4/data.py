import csv
import matplotlib.pyplot as plt

X1, X2, Y1, Y2 = [],[],[],[]

# open and format data 
with open('ex2data2.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    for row in csv_reader:
        if row["y"] == "1": 
            X1.append(float(row["x1"]))
            Y1.append(float(row["x2"]))
        else : 
            X2.append(float(row["x1"]))
            Y2.append(float(row["x2"]))

# # plot results
plt.plot(X1,Y1,'+', color="black", label = 'y = 1')
plt.plot(X2,Y2,'o', color="yellow", label = 'y = 0')
# plt.plot(X,Y,'o')
plt.xlabel("Microship Test 1")
plt.ylabel("Microship Test 2")
plt.legend()
plt.title("data plot")
plt.show()
from pocket import generateData
import numpy as np
import matplotlib.pyplot as plt

def Lss(S,w):
    ans = 0
    for i in range(len(S)):
        ans += (S[i][1] - w.T @ S[i][0])**2
    return ans/len(S)

def adaline(S, w, Tmax):
    for t in range(Tmax):
        for i in range(len(S)):
            if (S[i][1] - w.T @ S[i][0]) != 0 : 
                for j in range(len(w)):
                    w[j][0] += 2 * (S[i][1] - (w.T @ S[i][0])) * S[i][0][j]
    return (w,t, Lss(S,w))


S = generateData(100)
w = np.array([[0.76], [0.46], [0.26]])
wop, conv, t = adaline(S, w, 1)
print(wop, conv, t)

for e in S:
    if e[1] == 1 :plt.plot(e[0][1], e[0][2], "o", color = 'red')
    else :plt.plot(e[0][1], e[0][2], "*", color = 'green')
x = [0,30]
# y = [hyperplan(wop, x[0]),hyperplan(wop, x[1])]
# plt.plot(x, y,'-b')
plt.show()

from data import *
from perceptron import *
# from adaline import *
from pocket import *

S_nsep = generateData(100,0,15,10,20,0,15,10,20)
S_sep = generateData(100,0,10,10,20,0,10,10,20)

w0 = np.array([[0.76], [0.46], [0.26]])

# wop1, convPLA, t1 = PLA(S_sep, w0, hyperplan)
# print("w.T =", wop1.T,"| Ls(w) =", convPLA, "| t =", t1)

wop2, convPocket, t2 = Pocket(S_nsep, w0, 50, hyperplan)
print("w.T =", wop2,"| Ls(w) =", convPocket, "| t =", t2)

# wop3, convAdaline, t3 = adaline(S_nsep, w0, 50)
# print("w.T =", wop3,"| Ls(w) =", convAdaline, "| t =", t3)
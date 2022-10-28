from data import *
from perceptron import *
from adaline import *
from pocket import *
import time

S_nsep = generateData(200,0,15,5,20,0,15,5,20)
S_sep = generateData(100,0,10,10,20,0,10,10,20)
S_3D_sep = generateData3D(100,[0,9,11,20],[0,9,11,20],[0,9,11,20])
S_3D_nsep = generateData3D(200,[0,15,5,20],[0,15,5,20],[0,15,5,20])

w0 = np.array([[0.76], [0.46], [0.26], [0.11]])

# print("Perceptron Learning Algorithm:")
# startTime = time.time()
# wop1, convPLA, t1, t1_list, conv1_list, w1_list = PLA(S_3D_sep, w0, hyperplan3D, viz_mode="3d", visualize=True, save = True)
# endTime = time.time()
# print("w.T =", wop1.T,"| Ls(w) =", convPLA, "| t =", t1)
# PLATime = endTime - startTime
# # print("t1_list= ", t1_list)
# # print("conv1_list= ", conv1_list)
# print("Execution time: ", PLATime)

print("\nPocket Algorithm:")
startTime = time.time()
wop2, convPocket, t2 = Pocket(S_3D_nsep, w0, 100, hyperplan3D, viz_mode="3d", visualize=True)
endTime = time.time()
print("w.T =", wop2,"| Ls(w) =", convPocket, "| t =", t2)
PocketTime = endTime - startTime
# print("t2_list= ", t2_list)
# print("conv2_list= ", conv2_list)
print("Execution time: ", PocketTime)


# print("\nAdaline Algorithm:")
# startTime = time.time()
# wop3, convAdaline, t3 = adaline(S_3D_sep, w0, 300, hyperplan3D, visualize=True, viz_mode="3d")
# endTime = time.time()
# print("w.T =", wop3,"| Ls(w) =", convAdaline, "| t =", t3)
# AdalineTime = endTime - startTime
# # print("t3_list= ", t3_list)
# # print("conv3_list= ", conv3_list)
# print("Execution time: ", AdalineTime)

# Comparing the execution time of each algorithm
# bar = {'Pocket Algorithm':PocketTime, 'Adaline Algorithm':AdalineTime}
# algorithms = list(bar.keys())
# duration = list(bar.values())
# plt.bar(algorithms, duration, color ='green')
# plt.xlabel("Algorithms")
# plt.ylabel("Execution time in seconds")
# plt.title("Comparison of the execution time of each algorithm.")
# plt.show()

# # Comparing the emperical error of each algorithm
# plt.plot(t1_list, conv1_list,'-b', label="Perceptron")
# plt.plot(t2_list, conv2_list,'-r', label="Pocket")
# # plt.plot(t3_list, conv3_list,'-g', label="Adaline")
# plt.legend()
# plt.xlabel("Iterations")
# plt.ylabel("Emperical Error")
# plt.title("Comparison of the emperical error evolution of each algorithm.")
# plt.show()



# import matplotlib.animation as animation

# # fig, ax = plt.subplots()

# # x = [0,20]
# # y = [hyperplan(wop, x[0]),hyperplan(wop, x[1])]
# # line, = ax.plot(x, y)

# def calc_y(w_list, x, hyperplan):
#     return [[hyperplan(w, x[0]), hyperplan(w, x[1])] for w in w_list]


# def init():
#     line.set_data([],[])
#     return line

# def animate(i):
#     line.set_data(x,Y[i])  # update the data.
#     # line.set_data(x[1],Y[i][1])  # update the data.
#     return line,


# fig = plt.figure()
# ax = plt.axes(xlim=(0, 20), ylim=(0, 20))
# x = [0,20]
# Y = calc_y(w2_list, x,hyperplan)
# line, = ax.plot([], [])
# print(Y)
# print(w2_list)

# ani = animation.FuncAnimation( fig, animate,init_func=init, frames=len(Y), interval=1000)
# plt.draw()
# plt.show()
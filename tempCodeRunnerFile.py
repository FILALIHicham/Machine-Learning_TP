on Learning Algorithm:")
startTime = time.time()
wop1, convPLA, t1, t1_list, conv1_list, w1_list = PLA(S_3D_sep, w0, hyperplan3D, viz_mode="3d", visualize=True, save = True)
endTime = time.time()
print("w.T =", wop1.T,"| Ls(w) =", convPLA, "| t =", t1)
PLATime = endTime - startTime
# print("t1_list= ", t1_list)
# print("conv1_list= ", conv1_list)
print("Execution time
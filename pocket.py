from perceptron import PLA, Ls

def Pocket(S, w, Tmax, hyperplan):
    w = PLA(S, w, hyperplan)[0]
    for t in range(Tmax):
        wtemp = PLA(S, w, hyperplan)[0]
        if Ls(S, w, len(S)) > Ls(S, wtemp, len(S)) : w = wtemp
    return (w, t, Ls(S, w, len(S)))
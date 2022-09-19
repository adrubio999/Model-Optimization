import numpy as np
# rec_signal recieves a signal with shape (10000, Ts) and returns a signal with shape (10000,). If any of the ts have a 1,
# the generated value for the window is a 1
def rec_signal(y):
    len=np.shape(y)[0]
    print(np.shape(y))
    r_signal=np.empty(shape=(len))
    for i,w in enumerate(y):
        if any(w)==1:
            r_signal[i]=1
    return r_signal
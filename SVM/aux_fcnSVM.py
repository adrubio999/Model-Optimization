import numpy as np
def rec_signal(y):
    len=np.shape(y)[0]
    print(np.shape(y))
    r_signal=np.empty(shape=(len))
    for i,w in enumerate(y):
        if any(w)==1:
            r_signal[i]=1
    return r_signal
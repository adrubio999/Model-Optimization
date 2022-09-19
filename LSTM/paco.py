import pickle
import numpy as np
with open('C:\ProyectoInicial\Datos_pickle\\x_Dlx1.pickle', 'rb') as handle:
    x=pickle.load(handle)[:,[1,3,5]]
print(np.shape(x))
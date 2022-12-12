import pickle
from time import time
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

# Load data (deserialize)




arqs=['XGBOOST','LSTM','CNN2D','CNN1D']

fig,axs=plt.subplots(2,len(arqs))
for a,arq in enumerate(arqs):
    train_loss=[]
    if arq=='SVM':
        continue
    Root=f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Figure 2\{arq}\\'
    sp=[]
    for filename in os.listdir(Root+'Results'):
        f = os.path.join(Root+'Results', filename)
        # Creación de array con las características de cada modelo
        print(filename)
        if not(sp):
            sp=filename.split('_')[1:]
            params_arr=np.array(sp)
        else:
            sp=filename.split('_')[1:]
            params_arr=np.vstack([params_arr,sp])
        #print(params_arr)
        #input("Paise")
        

        # Acceso a loss
        with open(f, 'rb') as handle:
            results=(pickle.load(handle))['results']
        if arq!='XGBOOST':
            train_loss.append(results["train_losses"])
        else:
            train_loss.append(results["train_losses"]['logloss'])
    # Fila 1 del subplot: características de cada modelo
    unique_params_arr=[]
    for i in range(params_arr.shape[1]):
        unique_params=np.unique(params_arr[:,i])
        for j,param in enumerate(unique_params):
            params_arr[params_arr[:,i]==param,i]=j
    params_arr=np.array(params_arr,dtype=int)
    for params in params_arr:
        #input(params)
        print(np.linspace(1,params_arr.shape[1],params_arr.shape[1]))
        axs[0,a].plot(np.linspace(1,params_arr.shape[1],params_arr.shape[1],dtype=int),params,'.-')
    axs[0,a].set_yscale('linear')

    # Fila 2 del subplot: loss functions
    for loss in train_loss:
        axs[1,a].plot(loss)
    axs[1,a].set_title(arq)
    axs[1,a].set_yscale('log')
    
plt.show()

import pickle
from time import time
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization\PaperFigures')
from fig_aux_fcn import define_colors
# Load data (deserialize)


n_best_models=10
arqs=['XGBOOST','LSTM','CNN2D','CNN1D']

fig,axs=plt.subplots(2,len(arqs),tight_layout=True,figsize=(14,6))
for a,arq in enumerate(arqs):
    train_loss=[]
    F1_test_arr=[]
    if arq=='SVM':
        continue
    Root=f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Models\{arq}\\'
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
        

        # Acceso a loss
        with open(f, 'rb') as handle:
            results=(pickle.load(handle))['results']
        if arq!='XGBOOST':
            train_loss.append(results["train_losses"])
        else:
            train_loss.append(results["train_losses"]['logloss'])
        F1_test_arr.append(results["performance"][-2])
    
    # 10 best train F1
    colors,alpha_arr=define_colors(F1_test_arr,n_best_models)[:2]

    #input("BP")
    # Fila 1 del subplot: características de cada modelo
    unique_params_arr=[]
    n_params=params_arr.shape[1]
    for i in range(n_params):
        unique_params=np.unique(params_arr[:,i])
        for j,param in enumerate(unique_params):
            params_arr[params_arr[:,i]==param,i]=j
    params_arr=np.array(params_arr,dtype=int)
    for params,color,alpha in zip(params_arr,colors,alpha_arr):
        #input(params)
        axs[0,a].plot(np.linspace(1,n_params,n_params,dtype=int),params,'.-',c=color,alpha=alpha)

    axs[0,a].set_yscale('linear')
    axs[0,a].set_title(arq)

    # Fila 2 del subplot: loss functions
    for loss,color,alpha in zip(train_loss,colors,alpha_arr):
        axs[1,a].plot(loss,c=color,alpha=alpha)
    axs[1,a].set_yscale('log')
plt.show()

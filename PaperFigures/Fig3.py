import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys
import seaborn as sns
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import session
from fig_aux_fcn import define_colors,colors_dic

##############################################
# For the time being, this has to stay at 10
n_best_models=10

##############################################
blue=mpl.colormaps['PuBu']  
arqs=['XGBOOST','SVM','LSTM','CNN2D','CNN1D']
fig,axs=plt.subplots(3,len(arqs),figsize=(15,7),tight_layout=True)
n_sessions=21
session_names=[]
for s in range(n_sessions):
    session_names.append(session[s])
for a,arq in enumerate(arqs):
    Root=f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Models\{arq}\Validation'  # Load data (deserialize)
    Model=[]
    Codes=[]
    for filename in os.listdir(Root):
        Codes.append(filename[8:])
        f = os.path.join(Root, filename)
        # checking if it is a file
        with open(f, 'rb') as handle:
            Model.append(pickle.load(handle))
    ############################################
    ############################################  
    n_models=len(Codes)
    F1_max_color=np.empty(shape=n_models)
    ####################################################################################
    # COLOR DEFNITION
    for n in range(n_models):
        performance=np.nan_to_num(Model[n]['Performance'])
        n_sessions,n_th,j=np.shape(performance)
        F1_arr=np.empty(shape=(n_sessions,n_th))
        # Solo hay un array de ths
        for i in range(n_sessions):
            F1_arr[i]=performance[i,:,4]
        F1_max_color[n]=np.max(np.mean(F1_arr,axis=0))    # max de la media de un model (de todas las sesiones, para cada th) de los F1
    colors,_,best_index=define_colors(F1_max_color,n_best_models,arq)
    # Saving best model

    with open(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModels\{arq}.pickle', 'wb') as handle:
        pickle.dump(Model[best_index[0]], handle, protocol=pickle.HIGHEST_PROTOCOL)
    #Model reordering: Max F1 -> Min F1
    Model=[Model[i] for i in best_index]
    ###############################################################################
    prec_means=[]
    rec_means=[]
    F1_means=[]
    F1_95=[]
    F1_05=[]
    th_arrays=[]
    F1_arr_best_mean_th=[]
    # Para la matriz de calor
    F1_max=np.empty(shape=(n_models,n_sessions))
    # 1st row: P-R
    for n in range(n_models):
        print(Codes[n])
        performance=np.nan_to_num(Model[n]['Performance'])
        n_sessions,n_th,j=np.shape(performance)
        th_arr=performance[0,:,1]
        prec_arr=np.empty(shape=(n_sessions,n_th))
        rec_arr=np.empty(shape=(n_sessions,n_th))
        F1_arr=np.empty(shape=(n_sessions,n_th))
        # Solo hay un array de ths
        for i in range(n_sessions):
            prec_arr[i]=performance[i,:,2]
            rec_arr[i]=performance[i,:,3]
            F1_arr[i]=performance[i,:,4]
            F1_max[n][i]=max(performance[i,:,4])

        prec_means.append(np.mean(prec_arr,axis=0))
        rec_means.append(np.mean(rec_arr,axis=0))
        F1_means.append(np.mean(F1_arr,axis=0))
        F1_95.append(np.percentile(F1_arr,95,axis=0))
        F1_05.append(np.percentile(F1_arr,5,axis=0))
        ind_max_mean_F1=np.argmax(np.mean(F1_arr,axis=0))

        F1_arr_best_mean_th.append(F1_arr[:,ind_max_mean_F1])

        th_arrays.append(th_arr)

        for Ps,Rs,color in zip(prec_arr,rec_arr,colors):
            axs[0,a].plot(Rs,Ps,'-',c=color,alpha=0.05)
        
        
        #for F1,color in zip(F1_arr,colors):
        #    axs[1,a].plot(th_arr,F1,'-',c=color,alpha=0.05)
    # F1_max[n,i]: maximo F1 para cada modelo n y sesión i
    # F1_means

    for i in range(n_models):
        axs[0,a].plot(rec_means[i],prec_means[i],'-',c=colors[i],marker='.')
        axs[1,a].plot(th_arrays[i],F1_means[i],'-',marker='.',c=colors[i]  )
        if i in best_index:
            axs[1,a].fill_between(th_arrays[i],F1_95[i],F1_05[i],color=colors[i],alpha=0.05)
    # Row 3

    F1_media_de_maximos=np.mean(F1_max,axis=1)
    F1_std_de_maximos=np.std(F1_max,axis=1)
    X=np.linspace(0,n_models-1,n_models,dtype=int)
    inc=1.0/n_sessions
    F1_arr_best_mean_th=np.array(F1_arr_best_mean_th)
    for j in range(n_sessions):
        axs[2,a].plot(X,F1_arr_best_mean_th[:,j],'.',c=[element*j*inc for element in colors_dic[arq]],alpha=0.6)

    #axs[2,a].bar(X,F1_media_de_maximos,color=colors_dic[arq],alpha=0.33)
    F1_max_de_medias=np.max(F1_means,axis=1)
    F1_std_mejor_th=np.std(F1_arr_best_mean_th,axis=1)
    axs[2,a].bar(X,F1_max_de_medias,color=colors,alpha=1)
    axs[2,a].errorbar(X,F1_max_de_medias,F1_std_mejor_th/2,c='k',elinewidth=2,ls='none')

    axs[0,a].set_title(arq)
    axs[0,0].set_ylabel('R')
    axs[0,0].set_xlabel('P')
    axs[1,0].set_ylabel('Threshold')
    axs[1,0].set_xlabel('F1')
    axs[1,0].get_shared_y_axes().join(axs[1,0],axs[1,1],axs[1,2],axs[1,3],axs[1,4])

    axs[2,0].set_ylabel("Mean F1")
    axs[2,0].get_shared_y_axes().join(axs[2,0],axs[2,1],axs[2,2],axs[2,3],axs[2,4])
plt.savefig('C:\\Users\Adrian\Desktop\\Paper\\Figura 3.svg')
plt.show()



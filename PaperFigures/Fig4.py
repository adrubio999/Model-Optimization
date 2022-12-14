import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys
import seaborn as sns
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import session
from fig_aux_fcn import define_colors
#############################################
Stability_prop=0.9

##############################################
blue=mpl.colormaps['PuBu']  
arqs=['XGBOOST','SVM','LSTM','CNN2D','CNN1D']
fig,axs=plt.subplots(2,3,figsize=(15,7),tight_layout=True)
n_sessions=21
n_best_models=5
session_names=[]
for s in range(n_sessions):
    session_names.append(session[s])
Model=[]

for a,arq in enumerate(arqs):
    with open(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModels\{arq}.pickle', 'rb') as handle:
        Model.append(pickle.load(handle))
    ##################################################3
    ############################################  
n_models=len(Model)
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
colors,_,best_index=define_colors(F1_max_color,n_best_models)

###############################################################################
prec_means=[]
rec_means=[]
F1_means=[]
F1_95=[]
F1_05=[]
th_arrays=[]
StI=[]
# Para la matriz de calor
F1_max=np.empty(shape=(n_models,n_sessions))
F1_max_plot=np.empty(shape=(n_models,n_sessions,3))
# 1st row: P-R
for n in range(n_models):
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
        F1_max_plot[n][i]=performance[i,np.argmax(performance[i,:,4]),2:]
    
    prec_means.append(np.mean(prec_arr,axis=0))
    rec_means.append(np.mean(rec_arr,axis=0))
    F1_means.append(np.mean(F1_arr,axis=0))
    F1_95.append(np.percentile(F1_arr,95,axis=0))
    F1_05.append(np.percentile(F1_arr,5,axis=0))
    th_arrays.append(th_arr)
    stability_val=np.max(F1_means[n])*Stability_prop
    StI.append( len(F1_means[n][F1_means[n]>=stability_val]) / n_th)

    for Ps,Rs,color in zip(prec_arr,rec_arr,colors):
        axs[0,0].plot(Rs,Ps,'-',c=color,alpha=0.05,label='_Paco')
    #for F1,color in zip(F1_arr,colors):
    #    axs[0,1].plot(th_arr,F1,'-',c=color,alpha=0.05)
# F1_max[n,i]: maximo F1 para cada modelo n y sesión i
# F1_means

for i in range(n_models):
    axs[0,0].plot(rec_means[i],prec_means[i],'-',c=colors[i],marker='.',label=arqs[i])
    axs[0,1].plot(th_arrays[i],F1_means[i],'-',marker='.',c=colors[i]  )
    if i in best_index:
        axs[0,1].fill_between(th_arrays[i],F1_95[i],F1_05[i],color=colors[i],alpha=0.05)

F1_mod_means=np.mean(F1_max,axis=1)
F1_mod_stdev=np.std(F1_max,axis=1)
X=np.linspace(0,n_models-1,n_models,dtype=int)
inc=1.0/n_sessions
for j in range(n_sessions):
    axs[0,2].plot(X,F1_max[:,j],'.',c=blue(j*inc))


#################################################
axs[0,0].legend()
axs[0,0].set_ylabel('R')
axs[0,0].set_xlabel('P')

axs[0,1].set_xlabel('Threshold')
axs[0,1].set_ylabel('F1')

axs[0,2].bar(X,F1_mod_means,alpha=0.33)
axs[0,2].errorbar(X,F1_mod_means,F1_mod_stdev/2,c=blue(0.8),elinewidth=1)
axs[0,2].set_xticks(np.arange(len(arq)))
axs[0,2].set_xticklabels(arqs,fontsize=10)
axs[0,2].set_ylabel("Mean F1")


#################################################
# Sub [1,0]: how the StI is computed            #
# ###############################################
stability_val=np.max(F1_means[0])*Stability_prop

axs[1,0].plot(th_arrays[0],F1_means[0],'-',c=colors[0],marker='.')
axs[1,0].axhline(stability_val,color=colors[0],alpha=0.3)
axs[1,0].set_xlabel('Threshold')
axs[1,0].set_ylabel('F1')
#################################################
#     Sub [1,1]: SbIplot                                 #
#################################################
axs[1,1].bar(X,StI,alpha=0.33)
axs[1,1].set_xticks(np.arange(len(arq)))
axs[1,1].set_xticklabels(arqs,fontsize=10)
axs[1,1].set_ylabel('Stability index')
#################################################
#     Sub [1,2]                                 #
#################################################

for i in range(n_models):
    axs[1,2].plot(StI[i],F1_mod_means[i],color=colors[i],marker='.')
axs[1,2].legend(arqs)
axs[1,2].set_xlabel('Stability index')
axs[1,2].set_ylabel('F1')
#axs[1,2].ylims(0,1)
#axs[1,2].xlims(0,1)
plt.show()
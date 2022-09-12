import pickle
import matplotlib.pyplot as plt

import numpy as np
import os
import seaborn as sns

# De donde se sacan los datos para comparar
Data_folder="Multicanal_Uds\\"
# Si se quieren guardar las figuras y donde. Si no se quiere guardar se queda cada figura en pantalla hasta que se pulse una tecla
SaveFig=False


if SaveFig==True:
    if not(os.path.exists('C:\Septiembre-Octubre\Optimizacion modelos\Pruebas LSTM\Plotting\\'+Data_folder)):
        os.makedirs('C:\Septiembre-Octubre\Optimizacion modelos\Pruebas LSTM\Plotting\\'+Data_folder)


###############################################
Root='C:\Septiembre-Octubre\Optimizacion modelos\Pruebas LSTM\\'+Data_folder+'Validation'

# Load data (deserialize)
Model=[]
Codes=[]
for filename in os.listdir(Root):
    print(filename)
    Codes.append(filename[8:])
    f = os.path.join(Root, filename)
    # checking if it is a file
    with open(f, 'rb') as handle:
        Model.append(pickle.load(handle))
##################################################3
############################################  
n_sessions=6
n_models=len(Codes)
prec_means=[]
rec_means=[]
F1_means=[]
th_arrays=[]
# Para la matriz de calor
F1_max=np.empty(shape=(n_models,n_sessions))
F1_max_plot=np.empty(shape=(n_models,n_sessions,3))

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
        F1_max_plot[n][i]=performance[i,np.argmax(performance[i,:,4]),2:]

    prec_means.append(np.mean(prec_arr,axis=0))
    rec_means.append(np.mean(rec_arr,axis=0))
    F1_means.append(np.mean(F1_arr,axis=0))
    th_arrays.append(th_arr)
            
# Plot de P y R, F1 y umbral
fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 6))
inc=1.0/n_models
for i in range(n_models):
    axs[0].plot(rec_means[i],prec_means[i],'-',marker='.')#,c=str(i*inc))
    axs[1].plot(th_arrays[i],F1_means[i],'-',marker='.')#,c=str(i*inc))

axs[0].set(xlabel="Recall",ylabel="Precision")
axs[1].set(xlabel="Threshold",ylabel="F1")
axs[0].legend(Codes,fontsize=10,loc='upper right')
axs[1].legend(Codes,fontsize=10,loc='upper right')
if SaveFig==True:
    fig.savefig('C:\Septiembre-Octubre\Optimizacion modelos\Pruebas LSTM\Plotting\\'+Data_folder+'Prec rec y F1 todas las arquitecturas.svg')
else:
    fig.waitforbuttonpress()

plt.close()
# Mean of models
F1_mod_means=np.mean(F1_max,axis=1)
F1_mod_stdev=np.std(F1_max,axis=1)
X=np.linspace(0,n_models-1,n_models,dtype=int)
for j in range(n_sessions):
    plt.plot(X,F1_max[:,j],'.')
plt.bar(X,F1_mod_means,alpha=0.33)
plt.errorbar(X,F1_mod_means,F1_mod_stdev/2,linestyle='--',elinewidth=1)
plt.ylabel("F1")
plt.xticks(X,Codes,rotation='vertical')
if SaveFig==True:
    plt.savefig('C:\Septiembre-Octubre\Optimizacion modelos\Pruebas LSTM\Plotting\\'+Data_folder+"Mean of models todas las arquitecturas.svg")
else:
    plt.waitforbuttonpress()
plt.close()

#Matriz de calor
F1_mat=np.empty(shape=(n_models,n_sessions))
for j in range(n_models):
    for i in range(n_sessions):
        F1_mat[j][i]=F1_max[j,i]
ax = sns.heatmap(F1_mat, linewidth=0.5,cbar_kws={'label': 'F1'})
ax.set(xlabel='Sessions', ylabel='Models')
ax.set_yticklabels(Codes,rotation=0)
if SaveFig==True:
    plt.savefig('C:\Septiembre-Octubre\Optimizacion modelos\Pruebas LSTM\Plotting\\'+Data_folder+"Heatmap todas las arquitecturas.svg")
else:
    plt.waitforbuttonpress()





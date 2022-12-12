import pickle
import matplotlib.pyplot as plt

import numpy as np
import os
import seaborn as sns
from aux_fcn import session
##############################################
# Si se quieren guardar las figuras y donde. Si no se quiere guardar se queda cada figura en pantalla hasta que se pulse una tecla
SaveFig=False
svg=True
saveBestModel=False
n_hf=5
n_lf=4
Test='\Band_pass2'
# De donde se sacan los datos para comparar
##############################################

Root='C:\Septiembre-Octubre\Model-Optimization\\Filters'+Test


# Where to save the generated figures-> Model type\ Plotting\ Test Name
FigFolder='C:\Septiembre-Octubre\Model-Optimization\\Filters\Plotting\\'
if SaveFig==True:
    if not(os.path.exists(FigFolder)):
        os.makedirs(FigFolder)
# Load data (deserialize)
Model=[]
Codes=[]
for filename in os.listdir(Root):
    if filename[0]!='R':
        continue
    print(filename)
    Codes.append(filename[8:])
    f = os.path.join(Root, filename)
    # Extracting the number of hf and lf bands
    


    # checking if it is a file
    with open(f, 'rb') as handle:
        Model.append(pickle.load(handle))
############################################
############################################  
# color creation
colors=[]
offset=0.45
inc=(1.0-offset)/(n_lf)

# In this case, this aproach is valid (3 hf cutoffs)
for n_h in range (n_hf):
    for n_l in range(n_lf):
        if n_h<3:
            aux=[0,0,0]
            aux[n_h]=inc*n_l+offset
        else:
            aux=[inc*n_l+offset]*3
            aux[n_h-3]=0
        colors.append(tuple(aux))

colors.append((0,0,0)) # Color for no filter
input(colors)
print(len(colors))
print(f'N models: {len(Codes)}')


n_sessions=21
session_names=[]
for s in range(n_sessions):
    session_names.append(session[s])
print(session_names)

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
fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 5))
# Personalized colors
inc=1.0/n_models
for i in range(n_models):
    axs[0].plot(rec_means[i],prec_means[i],'-',marker='.',alpha=0.8,c=colors[i])
    axs[1].plot(th_arrays[i],F1_means[i],'-',marker='.',alpha=0.8,c=colors[i])

axs[0].set(xlabel="Recall",ylabel="Precision")
axs[1].set(xlabel="Threshold",ylabel="F1")
axs[1].legend(Codes,fontsize=6,loc='upper right')
# axs[1].legend(Codes,fontsize=10,loc='upper right')
if SaveFig==True:
    fig.savefig(FigFolder+'Prec rec y F1 todos los modelos.'+(('svg') if svg else ('png')))
else:
    fig.waitforbuttonpress()

plt.close()
plt.cla()
plt.clf()
# Mean of models
F1_mod_means=np.mean(F1_max,axis=1)
F1_mod_stdev=np.std(F1_max,axis=1)
X=np.linspace(0,n_models-1,n_models,dtype=int)
plt.figure(figsize=(10,5))
for j in range(n_sessions):
    plt.plot(X,F1_max[:,j],'.')
plt.bar(X,F1_mod_means,alpha=0.33)
plt.errorbar(X,F1_mod_means,F1_mod_stdev/2,linestyle='--',elinewidth=1)
plt.ylabel("F1")
plt.xticks(X,Codes,rotation='vertical')
if SaveFig==True:
    plt.savefig(FigFolder+"Mean of models todos los modelos."+(('svg') if svg else ('png')))
else:
    plt.waitforbuttonpress()
plt.close()
plt.cla()
plt.clf()

# HeatMap
F1_mat=np.empty(shape=(n_models,n_sessions))
for j in range(n_models):
    for i in range(n_sessions):
        F1_mat[j][i]=F1_max[j,i]
ax = sns.heatmap(F1_mat, linewidth=0.5,cbar_kws={'label': 'F1'})
ax.set(xlabel='Sessions', ylabel='Models')
ax.set_yticklabels(Codes,rotation=0)
ax.set_xticklabels(session_names,rotation=90)
if SaveFig==True:
    plt.savefig(FigFolder+"Heatmap todos los modelos."+(('svg') if svg else ('png')))
else:
    plt.waitforbuttonpress()
plt.close()
plt.cla()
plt.clf()

if saveBestModel:
    F1_np=np.array(F1_means)
    print(F1_np.shape)
    print(F1_np)
    # Returns the indexes of the max value. Im intersted in the session itself, but i'll save the th too.

    #model_ind,th_ind=np.where(F1_np==F1_np.max())
    model_ind=[9]
    th_ind=[7]
    print(model_ind,th_ind)
    print(F1_np[9].max(),F1_np[model_ind,th_ind])
    Code=Codes[model_ind[0]]
    input(Code)
    

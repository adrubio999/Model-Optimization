import pickle
from pyexpat.errors import codes
from unittest import result
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from aux_fcn import session,session_path
import shutil

SaveFig=False
svg=False
# If you want to move events from the data folder to a specific folder named Best_mmodels, to proceed with the ripple
# properties analysis in Matlab
MovEvents=False
Root='C:\Septiembre-Octubre\Model-Optimization\Best_models'
FigFolder=Root+'\Ploting'
if SaveFig==True:
    if not(os.path.exists(FigFolder)):
        os.makedirs(FigFolder)
# Load data (deserialize)
model=[]
Codes=[] # Codes is what will appear as legend in the plots, not the parameters
type_arr=[]
test_name_arr=[]
for filename in os.listdir(Root):
    print(filename)
    # checking if it is a file
    f = os.path.join(Root, filename)
    with open(f, 'rb') as handle:
        load_dic=pickle.load(handle)
    type_arr.append(load_dic["type"])
    test_name_arr.append(load_dic["test_name"])
    Codes.append(load_dic["code"])
    model.append(load_dic["results"])
print(Codes)
n_sessions=21
session_names=[]
for s in range(n_sessions):
    session_names.append(session[s])

n_models=len(Codes)
prec_means=[]
rec_means=[]
F1_means=[]
th_arrays=[]
# Para la matriz de calor
F1_max=np.empty(shape=(n_models,n_sessions))
F1_max_plot=np.empty(shape=(n_models,n_sessions,3))

for n in range(n_models):
    
    print(type_arr[n])
    performance=np.nan_to_num(model[n]['Performance'])
    print(performance.shape)
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

#############################################################################################

# Plot de P y R, F1 y umbral
fig, axs = plt.subplots(1, 2, constrained_layout=True, figsize=(15, 6))
inc=1.0/n_models
for i in range(n_models):
    axs[0].plot(rec_means[i],prec_means[i],'-',marker='.')#,c=str(i*inc))
    axs[1].plot(th_arrays[i],F1_means[i],'-',marker='.')#,c=str(i*inc))

axs[0].set(xlabel="Recall",ylabel="Precision")
axs[1].set(xlabel="Threshold",ylabel="F1")
axs[0].legend(type_arr,fontsize=10,loc='upper right')
axs[1].legend(type_arr,fontsize=10,loc='upper right')
if SaveFig==True:
    fig.savefig(FigFolder+'Prec rec y F1 todos los modelos.'+(('svg') if svg else ('png')))
else:
    fig.waitforbuttonpress()

plt.close()
plt.cla()
plt.clf()

#############################################################################################

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
plt.xticks(X,type_arr,rotation='vertical')
if SaveFig==True:
    plt.savefig(FigFolder+"Mean of models todos los modelos."+(('svg') if svg else ('png')))
else:
    plt.waitforbuttonpress()
plt.close()
plt.cla()
plt.clf()

#############################################################################################

# HeatMap
F1_mat=np.empty(shape=(n_models,n_sessions))
for j in range(n_models):
    for i in range(n_sessions):
        F1_mat[j][i]=F1_max[j,i]
ax = sns.heatmap(F1_mat, linewidth=0.5,cbar_kws={'label': 'F1'})
ax.set(xlabel='Sessions', ylabel='Models')
ax.set_yticklabels(type_arr,rotation=0)
ax.set_xticklabels(session_names,rotation=90)
if SaveFig==True:
    plt.savefig(FigFolder+"Heatmap todos los modelos."+(('svg') if svg else ('png')))
else:
    plt.waitforbuttonpress()

plt.close()
plt.cla()
plt.clf()
for filename in os.listdir(Root):
    print(filename)
for i in range(n_models):
    events_filename=type_arr[i]+'_'+test_name_arr[i]+'_'+Codes[i]+'_th'
    input(events_filename)
    for s in range (n_sessions):
        events_path=session_path[s]+'\events\\'+type_arr[i]+'\\'
        print(events_path)
        for filename in os.listdir(events_path):
            if events_filename==filename[:len(events_filename)]:
                print(events_path+filename)
                shutil.copyfile(events_path+filename,session_path[s]+'\events\\Best\\'+type_arr[i]+'_th_'+filename[len(events_filename):]) 
    

import pickle
from time import time
import numpy as np
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization\PaperFigures')
from aux_fcn import fcn_load_pickle,fcn_save_pickle
from fig_aux_fcn import add_dispersion,define_colors_parametric_search,colors_dic
# Load data (deserialize)

dispersion_mag=0.15
arqs=['XGBOOST','SVM','LSTM','CNN2D','CNN1D']

params=[['Channels', 'Window\nsize','Time steps \nfor window','Tree depth','Lambda','G','L','Scale'],
        ['Channels', 'Window\nsize','Time steps \nfor window', 'Undersampler\nproportion'],
        ['Channels', 'Window\nsize','Time steps \nfor window', 'Bidirectional','Layers','Units', 'Epochs','Samples per training batch'],
        ['Channels', 'Window\nsize','Time steps \nfor window','Configuration','Epoch','Samples per training batch'],
        ['Channels', 'Window\nsize','Time steps \nfor window','Epoch','Samples per training batch']]
std_of_means_all=[]
dif_of_means_all=[]
params_tag_all=[]
plt.rc('xtick', labelsize=8)
fig,axs=plt.subplots(5,4,figsize=(12,7),tight_layout=True)
for a,arq in enumerate(arqs):
    std_of_means_arq=[]
    dif_of_means_arq=[]
    params_tag_arq=[]
    train_loss=[]
    F1_test_arr=[]
    
    Root=f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Models\{arq}\\'
    sp=[]
    for filename in os.listdir(Root+'Results'):
        f = os.path.join(Root+'Results', filename)
        # Creación de array con las características de cada modelo
        print(filename)
        if not(sp):
            sp=filename[0:-7]
            sp=sp.split('_')[1:]
            params_arr=np.array(sp)
        else:
            sp=filename[0:-7]
            sp=sp.split('_')[1:]
            params_arr=np.vstack([params_arr,sp])
        results=fcn_load_pickle(f)['results']
        F1_test_arr.append(results["performance"][-2])
    print(params_arr)
    F1_test_arr=np.array(F1_test_arr)
    colors=define_colors_parametric_search(F1_test_arr,arq)
    n_models,n_params=params_arr.shape
    i_p=0
    for i in range(n_params):
        unique_params=np.unique(params_arr[:,i])
        if len(unique_params)==1:
            continue
        F1_test_param=np.zeros(shape=(len(unique_params),n_models//len(unique_params)))
        for j,param in enumerate(unique_params):
            indexes=np.argwhere((params_arr[:,i]==param)==True).flatten()

            F1_test_param[j]=F1_test_arr[indexes]# To compute stdev, max-min

            used_colors=[colors[ind] for ind in indexes]
            for ii,F1 in enumerate (F1_test_param[j]):   
                axs[a][i_p].plot(param,F1,'.',c=used_colors[ii])

        F1_means=np.mean(F1_test_param,axis=1)
        F1_std=np.std(F1_test_param,axis=1)
        # Medida de influencia del parámetro en el modelo
        std_of_means_arq.append(np.std(F1_means))
        dif_of_means_arq.append(np.max(F1_means)-np.min(F1_means))
        params_tag_arq.append(params[a][i])
        # Cambio muy gordo en los colores cuando solo hay dos, voy a hacer un apaño
        colors_mean=define_colors_parametric_search(np.hstack([F1_test_arr,F1_means]),arq)[-len(F1_means):]
        axs[a][i_p].bar(unique_params,F1_means,color=colors_mean,alpha=0.8)
        axs[a][i_p].errorbar(unique_params,F1_means,F1_std/2,c='k')
        axs[a][i_p].set_xticks(unique_params,fontsize=4)
        axs[a][i_p].set_ylim([0,0.8])
        axs[a][i_p].set_title(params[a][i],fontsize=8)
        if i_p==0:
            axs[a][i_p].set_ylabel(arq,fontsize=10)
        i_p+=1
    std_of_means_all.append(std_of_means_arq)
    dif_of_means_all.append(dif_of_means_arq)
    params_tag_all.append(params_tag_arq)
save={"Std": std_of_means_all,
      "Dif":  dif_of_means_all,
      "Params": params_tag_all,
}
fcn_save_pickle('C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Supl4_parameter_F1_std.pickle',save)
print(save)
plt.savefig('C:\\Users\\Adrian\\Desktop\\Paper\\Intermedia estudio paramétrico.svg')
plt.show()



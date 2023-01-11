import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization\PaperFigures')

from aux_fcn import fcn_load_pickle
from fig_aux_fcn import colors_dic,define_colors_parametric_search
F1_std_pickle=fcn_load_pickle('C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Supl4_parameter_F1_std.pickle')['Std']
F1_dif_pickle=fcn_load_pickle('C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Supl4_parameter_F1_std.pickle')['Dif']
params_pickle=fcn_load_pickle('C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Supl4_parameter_F1_std.pickle')['Params']
arqs=['XGBOOST','SVM','LSTM','CNN2D','CNN1D']
fig,axs=plt.subplots(2,5,figsize=(14,6),tight_layout=True)

for a,ax in enumerate (axs[0]):
    F1_std=F1_std_pickle[a]
    #input(params_pickle[a])
    colors=define_colors_parametric_search(F1_std,arqs[a])
    ax.bar(range(len(F1_std)),F1_std,color=colors)
    ax.set_title(arqs[a])
    ax.set_xticks(range(len(F1_std)), params_pickle[a],fontsize=8,rotation=90)
    if a==0:
        ax.set_ylabel("Std of means\n(Relative effect\nof the parameter)")
for a,ax in enumerate (axs[1]):
    F1_dif=F1_dif_pickle[a]
    colors=define_colors_parametric_search(F1_dif,arqs[a])

    ax.bar(range(len(F1_dif)),F1_dif,color=colors)
    ax.set_title(arqs[a])
    ax.set_xticks(range(len(F1_dif)), params_pickle[a],fontsize=8,rotation=90)
    if a==0:
        ax.set_ylabel("Dif of means\n(Relative effect\nof the parameter)")

#f.set_ytitle("# of positive detections")
plt.savefig('C:\\Users\\Adrian\\Desktop\\Paper\\Suplementaria estudio paramétrico.svg')
plt.show()
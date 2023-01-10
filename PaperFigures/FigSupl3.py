import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization\PaperFigures')

from aux_fcn import fcn_load_pickle

F1_mat=fcn_load_pickle('C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Supl3_consensus_F1.pickle')

arqs=['XGBOOST','SVM','LSTM','CNN2D','CNN1D']
plt.plot(figsize=(10,10))

f=sns.heatmap(F1_mat,cmap='viridis', linewidth=0.5,cbar_kws={'label': 'F1'})

#f.set_ytitle("# of positive detections")
plt.ylabel("# of coincident detectors")
f.set_yticklabels([5,4,3,2,1])
f.set_xticklabels(["Best th -0.2","Best th -0.1","Best th","Best th +0.1", "Best th +0.2"],rotation=0,fontsize=8)
plt.savefig('C:\\Users\\Adrian\\Desktop\\Paper\\Suplementaria consenso.svg')
plt.show()
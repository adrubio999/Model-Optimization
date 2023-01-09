import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization\PaperFigures')

from aux_fcn import fcn_load_pickle,compute_precision_recall_events

all_preds=fcn_load_pickle('C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Supl2_all_preds.pickle')
TP_preds=fcn_load_pickle('C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Supl2_TP_preds.pickle')
arqs=['XGBOOST','SVM','LSTM','CNN2D','CNN1D']
F1_all_mat=np.zeros(shape=(len(arqs),len(arqs)))
F1_TP_mat=np.zeros(shape=(len(arqs),len(arqs)))

for i,preds_arr in enumerate(all_preds):
    for j,gt_arr in enumerate(all_preds):
        F1_sess=np.zeros(shape=(21))
        if i==j:
            continue
        for s in range(21):
            pred_ind=preds_arr[s]
            gt_ind=gt_arr[s]

            _,_,F1_sess[s],_,_,_=compute_precision_recall_events(pred_ind,gt_ind,0)
        F1_all_mat[i][j]=np.mean(F1_sess)

for i,preds_arr in enumerate(TP_preds):
    for j,gt_arr in enumerate(TP_preds):
        F1_sess=np.zeros(shape=(21))
        if i==j:
            continue
        for s in range(21):

            pred_ind=preds_arr[s]
            gt_ind=gt_arr[s]
            _,_,F1_sess[s],_,_,_=compute_precision_recall_events(pred_ind,gt_ind,0)
        F1_TP_mat[i][j]=np.mean(F1_sess)

fig,axs=plt.subplots(1,2,figsize=(10.66,4))



hm1=sns.heatmap(F1_all_mat, linewidth=0.5,ax=axs[0],cbar_kws={'label': 'F1'})
hm1.set_xticklabels(arqs)
hm1.set_yticklabels(arqs,rotation=0)
hm1.set_title('Mean F1, all detections')
hm2=sns.heatmap(F1_TP_mat, linewidth=0.5,ax=axs[1],cbar_kws={'label': 'F1'})
hm2.set_xticklabels(arqs)
hm2.set_yticklabels(arqs,rotation=0)
hm2.set_title('Mean F1, true positives')

plt.savefig('C:\\Users\Adrian\Desktop\\Paper\\Suplementaria 2.svg')

plt.show()
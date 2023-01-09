import numpy as np
import sys
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization\PaperFigures')

from aux_fcn import fcn_save_pickle,fcn_load_pickle,compute_precision_recall_events,get_predictions_index,prediction_parser,session,pyr


n_sessions=21

fs=1250

# I'll left out LSTMcte
Arquitectures=['XGBOOST','SVM','LSTM','CNN2D','CNN1D']
th_arr=[0.4,0.5,0.3,0.1,0.5]
F1_mat=np.zeros(shape=(5,5))

F1_arr=np.zeros(shape=(5,21))
for i in range(5):

    for s in range (21):
        print(f'Session: {s}')
        x=fcn_load_pickle(f'C:\ProyectoInicial\Datos_pickle\\x_{session[s]}.pickle')
        y=fcn_load_pickle(f'C:\ProyectoInicial\Datos_pickle\\y_{session[s]}.pickle') 
        y_bin=np.zeros(shape=(x.shape[0]))
        
        for a,arq in enumerate(Arquitectures):
            th=th_arr[a]+(i-2)*0.1
            if th<0.1:
                th=0.1 
            print(th,arq)
            model=fcn_load_pickle(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModelsDic\{arq}.pickle')
            
            y_pred=prediction_parser(model,arq,x,s)
            y_bin+=1*(y_pred>=th)
        for n_d in [1,2,3,4,5]:
            print("Number of detectors: "+str(n_d))
            y_cons=1*(y_bin>=n_d) # Greater than or equal
            y_cons=y_cons.reshape(-1,1)
            # Ground truth indexes
            y_gt_ind=get_predictions_index(y,0.7)
            
            y_pred_ind=get_predictions_index(y_cons,0.2)
            prec,rec,F1_arr[n_d-1,s],a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
    print(F1_arr.shape,F1_arr)
    F1_means_det=np.flip(np.mean(F1_arr,axis=1))
    print(F1_means_det)
    F1_mat[:,i]=F1_means_det
    print(F1_mat)

fcn_save_pickle('C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Supl3_consensus_F1.pickle',F1_mat)
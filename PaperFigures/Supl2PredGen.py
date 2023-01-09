import numpy as np
import os
import sys
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization\PaperFigures')

from aux_fcn import fcn_save_pickle,fcn_load_pickle,compute_precision_recall_events,get_predictions_index,prediction_parser,session,pyr


n_sessions=21

fs=1250

# I'll left out LSTMcte
Arquitectures=['XGBOOST','SVM','LSTM','CNN2D','CNN1D']
th_arr=[0.4,0.5,0.3,0.1,0.5]

# preds_list=[arq][sess]
all_preds_list=[]
all_TP_list=[]
# The numbers of models that have to detect a ripple for the consensual model to detect ripple
for a,arq in enumerate(Arquitectures):
    models_pred=[]
    models_TP=[]
    model=fcn_load_pickle(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModelsDic\{arq}.pickle')
    print(model)
    for s in range(n_sessions):
        print('\n'+ "Session number " +str(s)+ ' ' +session[s])
        x=fcn_load_pickle(f'C:\ProyectoInicial\Datos_pickle\\x_{session[s]}.pickle')
        y=fcn_load_pickle(f'C:\ProyectoInicial\Datos_pickle\\y_{session[s]}.pickle') 
        y_pred=prediction_parser(model,arq,x,s)

        # GT: to extract TP
        y_gt_ind=get_predictions_index(y,0.7)
        # Quizá se pueda adaptar esta para trabajar con enteros (nº de clasificadores que marcan el positivo. en lugar de decimales de 0 a 1)
        y_pred_ind=get_predictions_index(y_pred,th_arr[a])
        _,_,_,TP,_,_=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
        models_pred.append(y_pred_ind)


        index=np.where(TP)[0]

        models_TP.append(y_pred_ind[index])

    all_preds_list.append(models_pred)
    all_TP_list.append(models_TP)

fcn_save_pickle('Supl2_all_preds.pickle',all_preds_list)
fcn_save_pickle('Supl2_TP_preds.pickle',all_TP_list)



    


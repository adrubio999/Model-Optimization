import pickle
from time import time
import numpy as np
import os
import sys
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import compute_precision_recall_events,get_predictions_index,format_predictions,prediction_parser,session,pyr

TestName="Test2"
Root='C:\Septiembre-Octubre\Model-Optimization\Consensus\\'+TestName+'\\'
# If you want to save the generated signal of the model
save_signal=False
# If you want to save the generated events as a txt for ripple properties analysis
save_events=True

Dummy=False

# If you want to use the best threshold for each arquitecture, instead of looping over differents arrays
use_best_th = True

#########################################################################################
if save_signal==True:
  if not(os.path.exists(Root+ 'Signal')):
        os.makedirs(Root+ 'Signal')

if Dummy==False:
    n_sessions=21
else:
    n_sessions=2


# If you want to validate the original model
if not(os.path.exists(Root+ 'Validation')):
    os.makedirs(Root+ 'Validation')
fs=1250
# The th for each individual model
th_arr=[0.05 , 0.1 , 0.2 , 0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9]

n_detectors_arr=[1,2,3,4,5]
# I'll left out LSTMcte
Arquitectures=['LSTM','CNN2D','CNN1D','XGBOOST','SVM']
results=np.empty(shape=(n_sessions,len(n_detectors_arr),5))
# The numbers of models that have to detect a ripple for the consensual model to detect ripple
# 5 en total : CNN1D, CNN2D,SVM,XGBOOST,LSTM
if use_best_th==False:
    for th in th_arr:
        for s in range(n_sessions):
            print('\n'+ "Session number " +str(s)+ ' ' +session[s])
            with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
                x=pickle.load(handle)
            with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
                y=pickle.load(handle)    
            y_bin=np.zeros(shape=(x.shape[0]))

            for Arq in Arquitectures:
                with open('C:\Septiembre-Octubre\Model-Optimization\Consensus\Results\\'+Arq+'_best_model', 'rb') as handle:
                    model=(pickle.load(handle))

                y_pred=prediction_parser(model,x,s)
                y_bin+=1*(y_pred>=th)
            # y_bin contiene el número de detectores que superan el umbral en cada instante
            print(np.histogram(y_bin,bins=[0,1,2,3,4,5,6]))
            for i,n_detectors in enumerate(n_detectors_arr):
                print("Number of detectors: "+str(n_detectors))
                y_cons=1*(y_bin>=n_detectors) # Greater than or equal
                y_cons=y_cons.reshape(-1,1)
                # Ground truth indexes
                y_gt_ind=get_predictions_index(y,0.7)
                # Quizá se pueda adaptar esta para trabajar con enteros (nº de clasificadores que marcan el positivo. en lugar de decimales de 0 a 1)
                y_pred_ind=get_predictions_index(y_cons,0.2)
                prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
                #print(prec,rec,F1)
                results[s][i]=[s,n_detectors,prec, rec, F1]
        print(results)
        Validation_results={
            "Threshold":th,
            "Performance":results,
        }

        with open(Root+ 'Validation\Results_th'+str(th)+'.pickle', 'wb') as handle:
            pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    Best_arr_th = [0.4,0.05,0.5,0.4,0.55]
    for s in range(n_sessions):
        print('\n'+ "Session number " +str(s)+ ' ' +session[s])
        with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
            x=pickle.load(handle)
        with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
            y=pickle.load(handle)    
        y_bin=np.zeros(shape=(x.shape[0]))
        for n_th,Arq in enumerate(Arquitectures):
            print(n_th,Arq,Best_arr_th[n_th])
            with open('C:\Septiembre-Octubre\Model-Optimization\Consensus\Results\\'+Arq+'_best_model', 'rb') as handle:
                model=(pickle.load(handle))
            y_pred=prediction_parser(model,x,s)
            y_bin+=1*(y_pred>=Best_arr_th[n_th])
        # y_bin contiene el número de detectores que superan el umbral en cada instante
        print(np.unique(y_bin,return_counts=True))
        for i,n_detectors in enumerate(n_detectors_arr):
            print("Number of detectors: "+str(n_detectors))
            y_cons=1*(y_bin>=n_detectors) # Greater than or equal
            y_cons=y_cons.reshape(-1,1)
            # Ground truth indexes
            y_gt_ind=get_predictions_index(y,0.7)
            # Quizá se pueda adaptar esta para trabajar con enteros (nº de clasificadores que marcan el positivo. en lugar de decimales de 0 a 1)
            y_pred_ind=get_predictions_index(y_cons,0.2)
            prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
            #print(prec,rec,F1)
            results[s][i]=[s,n_detectors,prec, rec, F1]
    print(results)
    Validation_results={
        "Threshold": 0 ,
        "Performance": results ,
    }

    with open(Root + 'Validation\Results_best_th.pickle', 'wb') as handle:
        pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
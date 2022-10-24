import pickle
from time import time
import numpy as np
import os
import sys
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import compute_precision_recall_events,get_predictions_index,format_predictions,prediction_parser,session,pyr

TestName="CompilationTest"
Root='C:\Septiembre-Octubre\Model-Optimization\Consensus\\'+TestName+'\\'
# If you want to save the generated signal of the model
save_signal=False
# If you want to save the generated events as a txt for ripple properties analysis
save_events=True

Dummy=False

#########################################################################################
if save_signal==True:
  if not(os.path.exists(Root+ 'Signal')):
        os.makedirs(Root+ 'Signal')

if Dummy==False:
    tharr=np.linspace(0.05,1,20)
    n_sessions=21
else:
    tharr=np.linspace(0.25,1,4)
    n_sessions=2


# If you want to validate the original model
if not(os.path.exists(Root+ 'Validation')):
    os.makedirs(Root+ 'Validation')
fs=1250

th_arr=np.linspace(0.2,0.8,4)
th=0.5

# 5 en total : CNN1D, CNN2D,SVM,XGBOOST,LSTM
with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[0]+'.pickle', 'rb') as handle:
    x=pickle.load(handle)
Arquitectures=['CNN2D','CNN1D','XGBOOST','LSTM','LSTMcte','SVM']
y_bin=[]
for Arq in Arquitectures:

    with open('C:\Septiembre-Octubre\Model-Optimization\Consensus\Results\\'+Arq+'_best_model', 'rb') as handle:
        model=(pickle.load(handle))

    y_pred=prediction_parser(model,x,0)
    y_bin=y_bin+1*(y_pred>=th)
print(np.histogram(y_bin))

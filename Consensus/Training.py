import pickle
from time import time
import numpy as np
import os
import sys
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import compute_precision_recall_events,get_predictions_index,format_predictions,session,pyr

TestName="CompilationTest"
Root='C:\Septiembre-Octubre\Model-Optimization\Consensus\\'+TestName+'\\'
# If you want to save the generated signal of the model
save_signal=False
# If you want to save the generated events as a txt for ripple properties analysis
save_events=True

n_models=0
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
# 5 en total : CNN1D, CNN2D,SVM,XGBOOST,LSTM
Arquitectures=['CNN1D','CNN2D']
for Arquitecture in Arquitectures:
    '''with open('C:\Septiembre-Octubre\Model-Optimization\Consensus\Models\\'+Arquitecture+'best_model.pickle', 'rb') as handle:
        model=(pickle.load(handle))'''
    print('\n'+"Working with best "+Arquitecture+'...')
    # For for the sessions
    for s in range(n_sessions):
        print('\n'+ "Session "+session[s])
    # Carga de los datos de validación (las 6 sesiones que no he utilizado para entrenar)
    with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
        if n_channels==8:
            x=pickle.load(handle)
        elif n_channels==3:
            x=pickle.load(handle)[:, [0,pyr[s],7]]
        elif n_channels==4:             # 3 canales: primero, piramidal y último
            x=pickle.load(handle)[:, [0,pyr['Amigo2_1']-1,pyr['Amigo2_1'],7]]
        else:
            x=(pickle.load(handle)[:,pyr[s]]).reshape(-1,1)
    # ripples_ind contiene inicio y final de ripple
    with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
        y=pickle.load(handle)
        '''if Arquitecture=


with open(Root+'Results\Results_'+dic['Code']+'.pickle', 'rb') as handle:
    Params=(pickle.load(handle))

n_channels=Params['params']["N channels"]
timesteps=Params['params']['Time steps']

print("Validating model with %d channels and %02d timesteps"% (n_channels,timesteps))
# Carga del modelo que toque
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
model = keras.models.load_model(Root+'Models\\Model_'+dic['Code'], compile=False)
model.compile(loss="binary_crossentropy", optimizer=optimizer)
model = keras.models.load_model(Root+'Models\\Model_'+dic['Code']+'\\')
for s in range (n_sessions):
    print('\n'+ "Session "+session[s])
    # Carga de los datos de validación (las 6 sesiones que no he utilizado para entrenar)
    with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
        if n_channels==8:
            x=pickle.load(handle)
        elif n_channels==3:
            x=pickle.load(handle)[:, [0,pyr[s],7]]
        elif n_channels==4:             # 3 canales: primero, piramidal y último
            x=pickle.load(handle)[:, [0,pyr['Amigo2_1']-1,pyr['Amigo2_1'],7]]
        else:
            x=(pickle.load(handle)[:,pyr[s]]).reshape(-1,1)
    # ripples_ind contiene inicio y final de ripple
    with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
        y=pickle.load(handle)
    

    x=x.reshape(1,-1,n_channels)
    # Predicción
    predictions = model.predict(x, verbose=True)
    aux=np.copy(predictions)
    y_predict=np.empty(shape=(np.shape(y)[0]))
    step=np.shape(y)[0]//np.shape(aux)[1]
    print(step)

    for i,window in enumerate(aux[0,:,0]):
        y_predict[i*step:(i+1)*step]=window

    if save_signal==True:
        with open(Root+ '\Signal\y_pred_'+session[s], 'wb') as handle:
            pickle.dump(y_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    y_gt_ind=get_predictions_index(y,0.7)
    for i,th in enumerate(tharr):
        print('Threshold {:1.3f}'.format(th))
        y_pred_ind=get_predictions_index(y_predict,th)
        # s: session number. aux_fcn has dictionaries that assign the correct path
        if save_events:
            format_predictions(y_pred_ind,s,'\\CNN1D\CNN1D_'+TestName+'_'+dic['Code']+'_th'+str(th)+'.txt') 
        prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
        # Modelo, # th1, #th2, P,R y F1
        results[s][i]=[s,th,prec, rec, F1]

# Otro for con cada sesión?

Validation_results={
    "Params":Params['params'],
    "Performance":results,
}
with open(Root+ '\Validation\Results_'+dic['Code']+'.val', 'wb') as handle:
    pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

'''
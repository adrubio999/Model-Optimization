import pickle
from time import time
import numpy as np
import os
import sys
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import compute_precision_recall_events,get_predictions_index,format_predictions,session,pyr
from aux_fcnCNN2D import pickle_data
import utils as ut
OgModel=False
# Load data (deserialize)
TestName="Optimization_test"
Root='C:\Septiembre-Octubre\Model-Optimization\CNN2D\\'+TestName+'\\'
# If you want to save the generated signal of the model
save_signal=False
# If you want to save the generated events
save_events=True
# If you want to validate the original model
fs=1250
Best_models=[]
#Carga de mejores modelos
if OgModel==False:
    for filename in os.listdir(Root+'Results'):
        f = os.path.join(Root+'Results', filename)
        print(filename)
        if (filename[0]!='R'):
            break

        # checking if it is a file
        
        with open(f, 'rb') as handle:
            Saved=(pickle.load(handle))
        print(Saved['results']['performance'])
        F1_train=Saved['results']['performance'][3]
        F1_test=Saved['results']['performance'][6]
        # open the file in the write mode
        # Sólo se guardan en validación las que superen un determinado valor de train F1
        if F1_train>=0:
            Val={
                "Code": filename[8:-7],
                "F1 train": F1_train,
                "F1 test": F1_test,
                "F1 val ses Dlx1": -1,
                "F1 val ses Thy7": -1,
                "F1 val ses PV6": -1,
                "F1 val ses PV7xChR2": -1,
                "F1 val ses Thy9": -1,
                "F1 val ses Thy1GCam1": -1,
                "F1 val mean": -1,
                }
            Best_models.append(Val)
# Dummy es True si se desean hacer pruebas de compilación
Dummy=True
if Dummy==False:
    tharr=np.linspace(0.05,1,20)
else:
    tharr=np.linspace(0.25,1,4)

n_sessions=6
results=np.empty(shape=(n_sessions,len(tharr),5))
print(np.shape(results))


session_path={1:'C:\ProyectoInicial\Datos\Kilosort\Thy7\\2020-11-11_16-05-00',  # Val
        0:'C:\ProyectoInicial\Datos\Kilosort\Dlx1\\2021-02-12_12-46-54',  # Val
        2:'C:\ProyectoInicial\Datos\Kilosort\PV6\\2021-04-19_14-02-31',      #Val
        3:'C:\ProyectoInicial\Datos\Kilosort\PV7xChR2\\2021-05-18_13-24-33', #Val
        4:'C:\ProyectoInicial\Datos\Kilosort\Thy9\\2021-03-16_12-10-32',     #Val
        5:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-18_14-40-16',     #Val
        }

for dic in Best_models:

    print(dic['Code'])

    with open(Root+'Results\Results_'+dic['Code']+'.pickle', 'rb') as handle:
        Params=(pickle.load(handle))
    n_channels=Params['params']["N channels"]
    timesteps=Params['params']['Time steps']
    
    print("El modelo usa %d canales divididos en ventanas de %d muestras" % (n_channels,timesteps))
    
    # Carga del modelo que toque
    model = keras.models.load_model(Root+'Models\\Model_'+dic['Code']+'\\')
    for s in range (n_sessions):
        print('\n')
        # Carga de los datos de validación (las 6 sesiones que no he utilizado para entrenar)
        with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
            if n_channels==8:
                x=pickle.load(handle)
            elif n_channels==3:
                x=pickle.load(handle)[:, [0,pyr[s],7]]
            else:
                x=(pickle.load(handle)[:,pyr[s]]).reshape(-1,1)
        datapath = session_path[s]
        
        a, ripples_tags, a, a, a ,a = ut.load_data_pipeline(
            datapath,pickle_datapath=pickle_data[s], desired_fs=fs, window_seconds = 0.050,
            overlapping = False, zscore= True, binary = False)
        # ripples_ind contiene inicio y final de ripple
        ripples_ind=np.around(ripples_tags*fs)
        with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
            y=pickle.load(handle)
        x=x[:len(x)-len(x)%timesteps,:].reshape(-1,timesteps,n_channels,1)
        y=y[:len(y)-len(y)%timesteps,:].reshape(-1,1,1)

        y_predict= np.empty(shape=(y.shape[0],1,1))
        # Predicción antes de expandir las ventanas
        windowed_signal= model.predict(x,verbose=1)
        print(windowed_signal.shape)
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
        # y_predict tiene la respuesta del modelo a cada ventana extendida Ts muestras 
        print(y_predict.shape)
        if save_signal==True:
            with open(Root+ '\Signal\y_pred_'+session[s], 'wb') as handle:
                pickle.dump(y_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        performances=[]

        for i,th in enumerate(tharr):
            y_pred_ind=get_predictions_index(y_predict,th)
            if save_events:
                format_predictions(y_pred_ind,session_path[s]+'\events\\CNN2D'+TestName+dic['Code']+'_th'+str(th)+'.txt') 
            prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,ripples_ind,0)
            # Modelo, # th1, #th2, P,R y F1
            results[s][i]=[s,th,prec, rec, F1]
    # Otro for con cada sesión?
    Validation_results={
        "Params":Params['params'],
        "Performance":results,
    }
    with open(Root+ '\Validation\Results_'+dic['Code']+'.val', 'wb') as handle:
        pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
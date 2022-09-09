import pickle
from time import time
import numpy as np
import os
from tensorflow import keras
from aux_fcn import compute_precision_recall_events,get_predictions_index,format_predictions,session,pyr

# Load data (deserialize)
Root='C:\Septiembre-Octubre\Optimizacion modelos\Pruebas LSTM\\Multicanal_Uds\\'
save_signal=True
# create the csv writer
Best_models=[]
#Carga de mejores modelos
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
Dummy=False
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
save_signal=False

for dic in Best_models:

    print(dic['Code'])

    with open(Root+'Results\Results_'+dic['Code']+'.pickle', 'rb') as handle:
        Params=(pickle.load(handle))
    n_channels=Params['params']["N channels"]
    timesteps=Params['params']['Time steps']
    
    print("El modelo usa %d canales divididos en ventanas de %d muestras" % (n_channels,timesteps))
    # Carga del modelo que toque
    model = keras.models.load_model(Root+'Models\\Model_'+dic['Code'])
    for s in range (n_sessions):
        # Carga de los datos de validación (las 6 sesiones que no he utilizado para entrenar)
        with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
            if n_channels==8:
                x=pickle.load(handle)
            elif n_channels==3:
                x=pickle.load(handle)[:, [0,pyr[s],7]]
            else:
                x=(pickle.load(handle)[:,pyr[s]]).reshape(-1,1)
        print(np.shape(x))
        with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
            y=pickle.load(handle)
        x=x[:len(x)-len(x)%timesteps,:].reshape(-1,timesteps,n_channels)
        y=y[:len(y)-len(y)%timesteps,:].reshape(-1,1,1)
        print(np.shape(x))
        print(np.shape(y))
        # Predicción
        y_predict = model.predict(x,verbose=1)
        y_predict=y_predict.reshape(-1,1,1)
        if save_signal==True:
            with open(Root+ '\Signal\y_pred_'+session[s], 'wb') as handle:
                pickle.dump(y_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        performances=[]
        y_gt_ind=get_predictions_index(y,0.7)
        for i,th in enumerate(tharr):
            y_pred_ind=get_predictions_index(y_predict,th)
            format_predictions(y_pred_ind,session_path[s]+'\events\\'+'1_Umbral_'+dic['Code']+'_th'+str(th)+'.txt')
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
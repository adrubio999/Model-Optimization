import pickle
from time import time
import numpy as np
import os
import sys
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import compute_precision_recall_events,get_predictions_index,format_predictions,session,pyr,session_path
import utils as ut
# Load data (deserialize)
TestName="NoSplitTest"
Root='C:\Septiembre-Octubre\Model-Optimization\CNN2D\\'+TestName+'\\'
OgModel=True
# If you want to save the generated signal of the model
save_signal=False
# If you want to save the generated events as a txt for ripple properties analysis
save_events=False
# If you want to test the compilation
Dummy=False
n_models=1
###################################################


if Dummy==False:
    tharr=np.linspace(0.05,1,20)
    n_sessions=21
else:
    tharr=np.linspace(0.25,1,4)
    n_sessions=2
fs=1250
Best_models=[]
Sorted_models=[]
F1_test_arr=[]
results=np.empty(shape=(n_sessions,len(tharr),5))


if not(os.path.exists(Root+ 'Validation')):
    os.makedirs(Root+ 'Validation')
# Original model validation
if OgModel==True:
    model = keras.models.load_model('C:\Septiembre-Octubre\Model-Optimization\CNN2D\original_model\model_prob_vf.h5')
    timesteps=40
    for s in range (n_sessions):
        print('\n'+ "Session "+session[s])
        # Carga de los datos de validación (las 6 sesiones que no he utilizado para entrenar)
        with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
            x=pickle.load(handle)
        datapath = session_path[s]
        
        a, ripples_tags, a, a, a ,a = ut.load_data_pipeline(
            datapath,pickle_datapath='C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', desired_fs=fs, window_seconds = 0.050,
            overlapping = False, zscore= True, binary = False)
        # ripples_ind contiene inicio y final de ripple
        ripples_ind=np.around(ripples_tags*fs)
        with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
            y=pickle.load(handle)
        x=x[:len(x)-len(x)%timesteps,:].reshape(-1,timesteps,8,1)
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
            print('Threshold % 1.3f',(th))
            y_pred_ind=get_predictions_index(y_predict,th)
            if save_events:
                format_predictions(y_pred_ind,s,'\\CNN2D\CNN2D_'+TestName+'_OgModel_th'+str(th)+'.txt') 
            prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,ripples_ind,0)
            # Modelo, # th1, #th2, P,R y F1
            results[s][i]=[s,th,prec, rec, F1]
    # Otro for con cada sesión?
    Validation_results={
        "Performance":results,
    }
    with open(Root+ 'Validation\Results_Og_Model.val', 'wb') as handle:
        pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)



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
    Val={
        "Code": filename[8:-7],
        }
    Best_models.append(Val)
    F1_test_arr.append(F1_test)
# Dummy es True si se desean hacer pruebas de compilación
indexes=np.argsort(F1_test_arr)[len(Best_models)-n_models:] # I select the n best models



for ind in indexes:
    Sorted_models.append(Best_models[ind])
    print("Model with Code "+ Best_models[ind]['Code']+"and F1 test "+str(F1_test_arr[ind] ))
print('\n\n'+str(len(Sorted_models))+ ' models will be validated')
input("Press enter to proceed with the analysis, or Ctrl+C to abort.")


for dic in Sorted_models:

    print('\n'+"Validating model "+dic['Code']+'...')

    with open(Root+'Results\Results_'+dic['Code']+'.pickle', 'rb') as handle:
        Params=(pickle.load(handle))
    n_channels=Params['params']["N channels"]
    timesteps=Params['params']['Time steps']
    
    print("El modelo usa %d canales divididos en ventanas de %d muestras" % (n_channels,timesteps))
    
    # Carga del modelo que toque
    model = keras.models.load_model(Root+'Models\\Model_'+dic['Code']+'\\')
    for s in range (n_sessions):
        print('\n'+ "Session "+session[s])
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
            datapath,pickle_datapath='C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', desired_fs=fs, window_seconds = 0.050,
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
            print('Threshold % 1.3f',(th))
            y_pred_ind=get_predictions_index(y_predict,th)
            if save_events:
                format_predictions(y_pred_ind,s,'\\CNN2D\CNN2D_'+TestName+'_'+dic['Code']+'_th'+str(th)+'.txt') 
            prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,ripples_ind,0)
            # Modelo, # th1, #th2, P,R y F1
            results[s][i]=[s,th,prec, rec, F1]
    # Otro for con cada sesión?
    Validation_results={
        "Params":Params['params'],
        "Performance":results,
    }
    with open(Root+ 'Validation\Results_'+dic['Code']+'.val', 'wb') as handle:
        pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
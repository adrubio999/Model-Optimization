import numpy as np
import pickle

import os
import sys
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')
from metrics import compute_precision_recall_events
from aux_fcn import compute_precision_recall_events,get_predictions_index,format_predictions, get_predictions_index,session,pyr
from aux_fcnSVM import rec_signal
# Load data (deserialize)
TestName="FinalTest"
Root='C:\Septiembre-Octubre\Model-Optimization\SVM\\'+TestName+'\\'
# If you want to save the generated signal of the model
save_signal=False
# If you want to save the generated events as a txt for ripple properties analysis
save_events=True
# The models with a test F1 above the next threshold will be validated
Dummy=False
n_models=10
fs=1250
Best_models=[]
Sorted_models=[]
F1_test_arr=[]

if save_signal==True:
  if not(os.path.exists(Root+ 'Signal')):
        os.makedirs(Root+ 'Signal')

#Carga de mejores modelos
for filename in os.listdir(Root+'Results'):
    f = os.path.join(Root+'Results', filename)
    
    if (filename[0]!='R'):
        break

    # checking if it is a file
    
    with open(f, 'rb') as handle:
        Saved=(pickle.load(handle))

    F1_train=Saved['results']['performance'][3]
    F1_test=Saved['results']['performance'][6]
    print(F1_test)
    print("Model : " +filename[8:-7] +" is above the F1 threshold.")
    Val={
        "Code": filename[8:-7],
        }
    Best_models.append(Val)
    F1_test_arr.append(F1_test)

indexes=np.argsort(F1_test_arr)[len(Best_models)-n_models:] # I select the n best models

for ind in indexes:
    Sorted_models.append(Best_models[ind])
    print("Model with Code "+ Best_models[ind]['Code']+"and F1 test "+str(F1_test_arr[ind] ))
print('\n\n'+str(len(Sorted_models))+ ' models will be validated')
input("Press enter to proceed with the analysis, or Ctrl+C to abort.")

# Dummy es True si se desean hacer pruebas de compilación
if Dummy==False:
    tharr=np.linspace(0.05,1,20)
    n_sessions=21
else:
    tharr=np.linspace(0.25,0.75,2)
    n_sessions=2

results=np.empty(shape=(n_sessions,len(tharr),5))
print(np.shape(results))

for dic in Sorted_models:

    print('\n\n Validating model '+dic['Code']+'...')

    with open(Root+'Results\Results_'+dic['Code']+'.pickle', 'rb') as handle:
        Params=(pickle.load(handle))
    n_channels=Params['params']["N channels"]
    timesteps=Params['params']['Time steps']
    
    print("The model uses %d channels arranged in  %d samples window size" % (n_channels,timesteps))
    
    # Carga del modelo que toque
    with open(Root+'Models\\Model_'+dic['Code'], 'rb') as handle:
        clf=pickle.load(handle)
    for s in range (n_sessions):
        print('\n'+ "Session number " +str(s)+ ' ' +session[s])
        # Carga de los datos de validación (las 6 sesiones que no he utilizado para entrenar)
        with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
            if n_channels==8:
                x=pickle.load(handle)
            elif n_channels==3:
                x=pickle.load(handle)[:, [0,pyr[s],7]]
            else:
                x=(pickle.load(handle)[:,pyr[s]]).reshape(-1,1)
                    
        # Reshape to fit with the expected model input
        with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
            y=pickle.load(handle)
        x=x[:len(x)-len(x)%timesteps,:].reshape(-1,timesteps*n_channels)
        y=y[:len(y)-len(y)%timesteps,:].reshape(-1,1,1)

        y_predict= np.empty(shape=(y.shape[0],1,1))
        # Prediction before expanding the windows
        windowed_signal= clf.predict_proba(x)[:,1]
        print('Model output: ', windowed_signal.shape)
        
        for i,window in enumerate(windowed_signal):
            y_predict[i*timesteps:(i+1)*timesteps]=window
        # y_predict: after expanding the windows, to be compatible with perf array
        print(y_predict.shape)
        if save_signal==True:
            with open(Root+ '\Signal\y_pred_'+session[s], 'wb') as handle:
                pickle.dump(y_predict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        performances=[]
        y_gt_ind=get_predictions_index(y,0.7)
        for i,th in enumerate(tharr):
            print('Threshold {:1.3f}'.format(th))
            y_pred_ind=get_predictions_index(y_predict,th)
            # s: session number. aux_fcn has dictionaries that assign the correct path
            if save_events:
                format_predictions(y_pred_ind,s,'\\SVM\SVM_'+TestName+'_'+dic['Code']+'_th'+str(th)+'.txt') 
            prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
            # Modelo, # th1, #th2, P,R y F1
            results[s][i]=[s,th,prec, rec, F1]
    # Otro for con cada sesión?
    Validation_results={
        "Params":Params['params'],
        "Performance":results,
    }
    with open(Root+ '\Validation\Results_'+dic['Code']+'.val', 'wb') as handle:
        pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL) #'''
#Funciones
from distutils.command.config import config
from tabnanny import verbose
from turtle import shape
import numpy as np
import os
import sys
import pickle
import numpy as np
import pickle
import sklearn as sk
from sklearn import svm
from sklearn import calibration
import imblearn
from imblearn.under_sampling import RandomUnderSampler

from aux_fcnSVM import rec_signal

sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')


from aux_fcn import get_predictions_index,perf_array,split_data,compute_precision_recall_events,pyr

downsampled_fs=1250


with open('C:\ProyectoInicial\Datos_pickle\\x_Amigo2_1.pickle', 'rb') as handle:
    x_amigo=pickle.load(handle)
with open('C:\ProyectoInicial\Datos_pickle\\x_Som_2.pickle', 'rb') as handle:
    x_som=pickle.load(handle)
with open('C:\ProyectoInicial\Datos_pickle\\y_Amigo2_1.pickle', 'rb') as handle:
    y=pickle.load(handle)
with open('C:\ProyectoInicial\Datos_pickle\\y_Som_2.pickle', 'rb') as handle:
    y=np.append(y,pickle.load(handle)) 
y=np.reshape(y,(-1,1))


# Definición de pruebas lugar de almacenamiento
TestName="Optimization"
# Carpeta de la prueba
root='C:\Septiembre-Octubre\Model-Optimization\SVM\\'+TestName+'\\'
print(root)

if len(os.listdir(root))==0: #Está vacío, hay que crear
    os.mkdir(os.path.join(root, "Models"))
    os.mkdir(os.path.join(root, "Results"))    
    os.mkdir(os.path.join(root, "Data"))
    os.mkdir(os.path.join(root, "Validation"))

# Si Dummy==true, prueba reducida solo para funcionaiento
Dummy=True
if Dummy==False:
    tharr=np.linspace(0.05,1,20)
else:
    tharr=np.linspace(0.25,1,4)
    
# Llamada a creación de la NN (se crearán varios modelos, uno por cada
# sesión de entrenamiento) Dentro del bucle a partir de aquí
# Habrá varios bucles anidados según los parámetros que se modifiquen:
# 1: Nº de canales de entrada. Only even numbers are allowed in the CNN2D.
n_channels_arr=[8,3,1] 
# 2: Segundos de duración de ventana en que se dividen los datos para hacer separación en train y test
window_size_arr=[60]
# 3: Muestras en cada ventana temporal. Falla con 16. ¿Por qué?
timesteps_arr=[1]
# 4: Undersampler proportion: 0.1 means 1 true for each sample of the gt
undersampler_arr=[1,0.5,0.1]
# 7: Nº de épocas
# 8: Nº de batch
n_train_batch_arr=[2**5]
for n_channels in n_channels_arr:
    if n_channels==8:
        x=np.append(x_amigo,x_som)
    elif n_channels==3:             # 3 canales: primero, piramidal y último
        x=np.append(x_amigo[:, [0,pyr['Amigo2_1'],7]],x_som[:, [0,pyr['Som_2'],7]]) 
    elif n_channels==1:
        x=np.append(x_amigo[:,pyr['Amigo2_1']],x_som[:,pyr['Som_2']])
    x=np.reshape(x,(-1,n_channels))
    for window_size in window_size_arr:
        x_test_or,y_test_or,x_train_or,y_train_or=split_data(x,y,n_channels=n_channels,window_dur=window_size,fs=downsampled_fs,split=0.7)
        
        # Bucle for para probar time stamps (anchuras de la sliding window distinta)
        for timesteps in timesteps_arr:
            x_train=x_train_or[:len(x_train_or)-len(x_train_or)%timesteps,:].reshape(-1,timesteps*n_channels)
            y_train_aux=y_train_or[:len(y_train_or)-len(y_train_or)%timesteps,:].reshape(-1,timesteps)
            y_train=rec_signal(y_train_aux)

            x_test=x_test_or[:len(x_test_or)-len(x_test_or)%timesteps,:].reshape(-1,timesteps*n_channels)
            y_test_aux=y_test_or[:len(y_test_or)-len(y_test_or)%timesteps,:].reshape(-1,timesteps)
            y_test=rec_signal(y_test_aux)

            print("Entrada: ",x_train.shape," Salida: ",y_train.shape)

            # Array of posible undersampler proportions
            for undersampler_prop in undersampler_arr:
                print('\n Undersampler proportion :',undersampler_prop)
                rus = RandomUnderSampler(sampling_strategy=undersampler_prop)
                # Nº de instantes con ripple: 78588
                x_train_us, y_train_us = rus.fit_resample(x_train, y_train)
                # Nº de épocas
                print(x_train_us.shape,y_train_us.shape)

                print('Nº de canales: %d, Duración de la ventana: %d, Time steps: %d, Undersampler proportion: %1.2f' % (n_channels,window_size,timesteps,undersampler_prop))
                
                clf = sk.calibration.CalibratedClassifierCV(svm.LinearSVC()) 

                #Training 
                history=clf.fit(x_train_us, y_train_us)

                # Predicción. Devuelve una predicción por ventana
                test_signal = clf.predict_proba(x_test)[:,1]
                print(test_signal.shape)
                train_signal=clf.predict_proba(x_train)[:,1]
                #train_signal=np.random.rand(x_train.shape[0],1)
                print("Respuesta del modelo:", train_signal.shape)
                # Hay que generar una señal compatible con perf array, una salida para cada instante de tiempo
                y_train_predict=np.empty(shape=(x_train.shape[0]*timesteps,1,1))
                for i,window in enumerate(train_signal):
                    y_train_predict[i*timesteps:(i+1)*timesteps]=window
                print("Señal adaptada para computar predicciones", y_train_predict.shape)
                # test_signal=np.random.rand(x_test.shape[0],1)
                y_test_predict=np.empty(shape=(x_test.shape[0]*timesteps,1,1))
                for i,window in enumerate(test_signal):
                    y_test_predict[i*timesteps:(i+1)*timesteps]=window
                print("Señal adaptada para computar predicciones", y_test_predict.shape)
                print(train_signal[0:4],y_train_predict[:160])
                y_train_aux=y_train_aux.reshape(-1,1,1)
                y_test_aux=y_test_aux.reshape(-1,1,1)
                print(y_train_predict.shape,y_train_aux.shape)
                # La gt es la de y_train_aux, sin colapsar las ventanas aún.
                perf=perf_array(y_train_predict,y_train_aux,tharr)
                # Parámetros de mejor F1 para training set
                th=perf[np.argmax(perf[:,3]),0]

                # Métricas para train. La GT se extrae de la señal y_train_aux original, que aún no ha colapsado las ventanas
                ytrain_gt_ind=get_predictions_index(y_train_aux,th)
                ytrain_pred_ind=get_predictions_index(y_train_predict,th)
                
                best_performance=[]
                train_prec,train_rec,train_F1,a,b,c=compute_precision_recall_events(ytrain_pred_ind,ytrain_gt_ind,0)
                # Métricas para test
                ytest_gt_ind=get_predictions_index(y_test_aux,th)
                ytest_pred_ind=get_predictions_index(y_test_predict,th)

                test_prec,test_rec,test_F1,a,b,c=compute_precision_recall_events(ytest_pred_ind,ytest_gt_ind,0)
                best_performance=np.append(best_performance,[th,train_prec,train_rec,train_F1,test_prec,test_rec,test_F1,train_F1/test_F1])
                # Predicciones (lista)
                Pred_list=[th,ytrain_pred_ind,ytest_pred_ind]
                # Saving the model
                path_dir = root + "Models\\"
                pickle.dump(clf, open(path_dir+"Model_Ch"+str(n_channels)+"_W"+str(window_size)+"_Ts"+str(timesteps)+"_Us{:1.2f}".format(undersampler_prop), 'wb'))

                # Almaceno en un diccionario
                results = {
                "performance": best_performance,
                "predictions": Pred_list,
                }
                params={
                "N channels": n_channels,
                "Window size": window_size,
                "Time steps": timesteps,
                "Under sampler proportion": undersampler_prop,
                }
                to_save={'results': results,
                'params':params,
                }
                # Store data (serialize): un archivo para cada bucle de entrenamiento

                with open(root+ 'Results\Results_Ch%d_W%d_Ts%d_Us%1.2f.pickle' % (n_channels,window_size,timesteps,undersampler_prop), 'wb') as handle:
                    pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ################# Fin del bucle'''
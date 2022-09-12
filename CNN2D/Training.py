#Funciones
from turtle import shape
from tensorflow import keras
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
import numpy as np
import os
import sys
import pickle
import model_builders as mb
import utils as ut
from aux_fcnCNN2D import session, session_datapath, pickle_data 
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import get_predictions_index,perf_array,split_data,compute_precision_recall_events,pyr

downsampled_fs=1250
# Defino si quiero que se usen las dos sesiones extensas como datos o no
overlapping=0

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
TestName="Optimization_test"
# Carpeta de la prueba
root='C:\Septiembre-Octubre\Model-Optimization\CNN2D\\'+TestName+'\\'
print(root)

if len(os.listdir(root))==0: #Está vacío, hay que crear
    os.mkdir(os.path.join(root, "Models"))
    os.mkdir(os.path.join(root, "Results"))    
    os.mkdir(os.path.join(root, "Data"))

# Si Dummy==true, prueba reducida solo para funcionaiento
Dummy=True
if Dummy==False:
    tharr=np.linspace(0.05,1,20)
else:
    tharr=np.linspace(0.25,1,4)
    
# Llamada a creación de la NN (se crearán varios modelos, uno por cada
# sesión de entrenamiento) Dentro del bucle a partir de aquí
# Habrá varios bucles anidados según los parámetros que se modifiquen:
# 1: Nº de canales de entrada
n_channels_arr=[8] 
# 2: Segundos de duración de ventana en que se dividen los datos para hacer separación en train y test
window_size_arr=[60]
# 3: Muestras en cada ventana temporal
window_seconds=[62]
# 5: Nº de capas

layer_arr=range(2,3)
# 6: Nº de unidades
units_arr=range(5,6)
# 7: Nº de épocas
n_epochs_arr=[1]
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
        for timesteps in window_seconds:
            x_train=x_train_or[:len(x_train_or)-len(x_train_or)%timesteps,:].reshape(-1,timesteps,n_channels,1)
            y_train_aux=y_train_or[:len(y_train_or)-len(y_train_or)%timesteps,:].reshape(-1,timesteps,1)
            x_test=x_test_or[:len(x_test_or)-len(x_test_or)%timesteps,:].reshape(-1,timesteps,n_channels,1)
            y_test_aux=y_test_or[:len(y_test_or)-len(y_test_or)%timesteps,:].reshape(-1,timesteps,1)
            # Dimension adaptation. The input of the model has to be [timesteps, n_channels,1]
            y_train=np.zeros(shape=[x_train.shape[0],1])
            for i in range(y_train_aux.shape[0]):
                y_train[i]=1  if any (y_train_aux[i]==1) else 0
            print("Train Input and Output dimension", x_train.shape,y_train.shape)
            
            y_test=np.zeros(shape=[x_test.shape[0],1])
            for i in range(y_test_aux.shape[0]):
                y_test[i]=1  if any (y_test_aux[i]==1) else 0
            print("Test Input and Output Dimension", x_test.shape,y_test.shape)
            input_shape=x_test.shape[1:]

            for n_layers in layer_arr:
                # Nº de unidades por capa
                for n_uds in units_arr:
                    # Nº de épocas
                    for n_epochs in n_epochs_arr:
                        # Tamaño del batch: 32 a 512 exponencialmente 2048,4096,2**18
                        for n_train_batch in n_train_batch_arr:
                            print('Nº de canales: %d, Duración de la ventana: %d, Time steps: %d, Nº de capas: %d, Nº de unidades: %d,\n Nº de épocas: %d, Train Batch Size: %d' % (n_channels,window_size,timesteps,n_layers,n_uds,n_epochs,n_train_batch))
                           
                            model = mb.model_builder_prob(filters_Conv1 = 32, filters_Conv2 = 16, filters_Conv3=8, filters_Conv4 = 16,
                                            filters_Conv5 =16, filters_Conv6=8, input_shape = input_shape, 
                                            learning_rate  = 1e-5)

                            # Entrenamiento Lo comento de momento, no va bien
                            history=model.fit(x_train, y_train,shuffle=False, epochs=n_epochs,batch_size=n_train_batch,validation_data=(x_test,y_test), verbose=1)

                            # Predicción. Devuelve una predicción por ventana
                            test_signal = model.predict(x_test,verbose=1)
                            train_signal=model.predict(x_train,verbose=1)
                            #train_signal=np.random.rand(x_train.shape[0],1)
                            print("Respuesta del modelo:", train_signal.shape)
                            # Hay que generar una señal compatible con perf array, 
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
                            ytest_pred_ind=get_predictions_index(y_test_predict,th)
                            ytest_gt_ind=get_predictions_index(y_test_aux,th)
                            test_prec,test_rec,test_F1,a,b,c=compute_precision_recall_events(ytest_pred_ind,ytest_gt_ind,0)
                            best_performance=np.append(best_performance,[th,train_prec,train_rec,train_F1,test_prec,test_rec,test_F1,train_F1/test_F1])
                            # Predicciones (lista)
                            Pred_list=[th,ytrain_pred_ind,ytest_pred_ind]
                            # Y almaceno el modelo
                            directory = "Model_Ch"+str(n_channels)+"_W"+str(window_size)+"_Ts"+str(timesteps)+"_Bi"+"_L"+str(n_layers)+"_U"+( (str(n_uds)) if len(str(n_uds))!=1 else ("0"+str(n_uds))) +"_E"+str(n_epochs)+"_TB"+str(n_train_batch)
                            path_dir = root + "Models\\"
                            if not os.path.exists(directory):
                                os.mkdir(os.path.join(path_dir, directory))
                            model.save(path_dir + directory)

                            # Almaceno en un diccionario
                            results = {
                            "train_losses": history.history['loss'],
                            "test_losses": history.history['val_loss'],
                            "performance": best_performance,
                            "predictions": Pred_list,
                            }
                            params={
                            "N channels": n_channels,
                            "Window size": window_size,
                            "Time steps": timesteps,
                            "Layers": n_layers,
                            "Units": n_uds,
                            "Epochs": n_epochs,
                            "Batch size": n_train_batch,
                            }
                            to_save={'results': results,
                            'params':params,
                            }
                            # Store data (serialize): un archivo para cada bucle de entrenamiento
                            with open(root+ 'Results\Results_Ch%d_W%d_Ts%d_L%d_U%d_E%d_TB%d.pickle' % (n_channels,window_size,timesteps,n_layers,n_uds,n_epochs,n_train_batch), 'wb') as handle:
                                pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
                ################# Fin del bucle
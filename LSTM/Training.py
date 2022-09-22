#Funciones
from tensorflow import keras
from tensorflow.keras.initializers import GlorotUniform, Orthogonal
import numpy as np
import os
import pickle
import timeit
import sys
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')
from aux_fcn import get_predictions_index,perf_array,split_data,compute_precision_recall_events,pyr
def get_simple_LSTM(input_shape,lr=0.005,dropout_rate=0.2,n_layers=3,layer_size=20,seed=0,bidirectional=False):
    ''' Parameters
    ----------
    input_shape: length 2 tuple
        input dimensionality (int or None, 1)
    lr: float
        Adam optimizer learning rate
    dropout_rate: float in <0, 1>
        Dropout layer dropout fraction during training
    layer_sizes: list of int
        size of layers [LSTM, LSTM]
    seed: int
        random seed+i for layers where i is index of layers that can be seeded
    Returns
    -------
    tf.keras Model instance'''
    keras.backend.clear_session()

    # input layer
    inputs = keras.layers.Input(shape=input_shape)

    #LSTM layers
    if bidirectional==False:
        for i in range(n_layers):
            if i==0:
                x = keras.layers.LSTM(layer_size, return_sequences=True,
                                    kernel_initializer=GlorotUniform(seed=seed),
                                    recurrent_initializer=Orthogonal(seed=seed+1),
                                    )(inputs)
                x = keras.layers.Dropout(dropout_rate, seed=seed+2)(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(dropout_rate, seed=seed+3)(x)
            else:
                x = keras.layers.LSTM(layer_size, return_sequences=True,
                            kernel_initializer=GlorotUniform(seed=seed+i),
                            recurrent_initializer=Orthogonal(seed=seed+1+i),
                            )(x)
                x = keras.layers.Dropout(dropout_rate, seed=seed+2+i)(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(dropout_rate, seed=seed+3+i)(x)
    else: # Didirectional
        for i in range(n_layers):
            if i==0:
                x = keras.layers.Bidirectional(keras.layers.LSTM(layer_size, return_sequences=True,
                                    kernel_initializer=GlorotUniform(seed=seed),
                                    recurrent_initializer=Orthogonal(seed=seed+1),
                                    ) )(inputs)
                x = keras.layers.Dropout(dropout_rate, seed=seed+2)(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(dropout_rate, seed=seed+3)(x)
            else:
                x = keras.layers.Bidirectional(keras.layers.LSTM(layer_size, return_sequences=True,
                            kernel_initializer=GlorotUniform(seed=seed+i),
                            recurrent_initializer=Orthogonal(seed=seed+1+i),
                            ) )(x)
                x = keras.layers.Dropout(dropout_rate, seed=seed+2+i)(x)
                x = keras.layers.BatchNormalization()(x)
                x = keras.layers.Dropout(dropout_rate, seed=seed+3+i)(x)

    predictions = keras.layers.Dense(1, activation='sigmoid',kernel_initializer=GlorotUniform(seed=seed+13))(x)
    # Define model
    model = keras.models.Model(inputs=inputs,
                               outputs=predictions,
                               name='RippleTry')

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])

    return model
downsampled_fs=1250
# Defino si quiero que se usen las dos sesiones extensas como datos o no
class SaveBatchLoss(keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        batch_end_loss.append(logs['loss'])
# Loss del training
class SaveTestBatchLoss(keras.callbacks.Callback):
    def on_test_batch_end(self, batch, logs=None):
        test_end_loss.append(logs['loss'])

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
# Carpeta de la prueba
root='C:\Septiembre-Octubre\Model-Optimization\LSTM\\Plot_signal\\'
if len(os.listdir(root))==0: #Está vacío, hay que crear
    os.mkdir(os.path.join(root, "Models"))
    os.mkdir(os.path.join(root, "Results"))    
    os.mkdir(os.path.join(root, "Data"))

# Si Dummy==true, prueba reducida solo para funcionaiento
Dummy=False
if Dummy==False:
    tharr=np.linspace(0.1,1,10)
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
time_steps_array=[32]
# 4: Bidirecional o no
bi_arr=[0]
# 5: Nº de capas
layer_arr=range(2,3)
# 6: Nº de unidades
units_arr=range(10,11)
# 7: Nº de épocas
n_epochs_arr=[30]
# 8: Nº de batch
n_train_batch_arr=[2**8]
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
        for timesteps in time_steps_array:
            # Bi o no
            for bi in bi_arr:
                # Nº de capas
                for n_layers in layer_arr:
                    # Nº de unidades por capa
                    for n_uds in units_arr:
                        # Nº de épocas
                        for n_epochs in n_epochs_arr:
                            # Tamaño del batch: 32 a 512 exponencialmente 2048,4096,2**18
                            for n_train_batch in n_train_batch_arr:
                                print('Nº de canales: %d, Duración de la ventana: %d, Time steps: %d, Bidireccional:  %d, Nº de capas: %d, Nº de unidades: %d,\n Nº de épocas: %d, Train Batch Size: %d' % (n_channels,window_size,timesteps,bi,n_layers,n_uds,n_epochs,n_train_batch))
                                # Timeo para conocer cuanto tarda cada iteración
                                start = timeit.default_timer()
                                # Reshape de las entradas de train y test
                                x_train=x_train_or[:len(x_train_or)-len(x_train_or)%timesteps,:].reshape(-1,timesteps,n_channels)
                                y_train=y_train_or[:len(y_train_or)-len(y_train_or)%timesteps,:].reshape(-1,timesteps,1)
                                x_test=x_test_or[:len(x_test_or)-len(x_test_or)%timesteps,:].reshape(-1,timesteps,n_channels)
                                y_test=y_test_or[:len(y_test_or)-len(y_test_or)%timesteps,:].reshape(-1,timesteps,1)
                                print(np.shape(x_test))
                                print(np.shape(x_train))

                                model=get_simple_LSTM(input_shape=(timesteps,n_channels),lr=0.005,dropout_rate=0.2,n_layers=n_layers,layer_size=n_uds,seed=0,bidirectional=bi)
                                batch_end_loss = list()
                                # Entrenamiento
                                history=model.fit(x_train, y_train, epochs=n_epochs,batch_size=n_train_batch,validation_data=(x_test,y_test), verbose=1,callbacks=SaveBatchLoss())
                                # Evaluación (test loss)
                                test_end_loss=list()
                                #score = model.evaluate(x_test, y_test, verbose = 1,batch_size=n_train_batch,callbacks=SaveTestBatchLoss())
                                # Predicción
                                ytest_predict = model.predict(x_test,verbose=1)
                                ytrain_predict=model.predict(x_train,verbose=1)
                                # Reshape para que el análisis de resultados funcione
                                ytrain_predict=ytrain_predict.reshape(-1,1,1)
                                ytest_predict=ytest_predict.reshape(-1,1,1)
                                y_train=y_train.reshape(-1,1,1)
                                y_test=y_test.reshape(-1,1,1)

                                perf=perf_array(ytrain_predict,y_train,tharr)
                                # Parámetros de mejor F1 para training set
                                th=perf[np.argmax(perf[:,3]),0]

                                # Métricas para train
                                ytrain_pred_ind=get_predictions_index(ytrain_predict,th)
                                ytrain_gt_ind=get_predictions_index(y_train,th)
                                best_performance=[]
                                train_prec,train_rec,train_F1,a,b,c=compute_precision_recall_events(ytrain_pred_ind,ytrain_gt_ind,0)
                                # Métricas para test
                                ytest_pred_ind=get_predictions_index(ytest_predict,th)
                                ytest_gt_ind=get_predictions_index(y_test,th)
                                test_prec,test_rec,test_F1,a,b,c=compute_precision_recall_events(ytest_pred_ind,ytest_gt_ind,0)
                                best_performance=np.append(best_performance,[th,train_prec,train_rec,train_F1,test_prec,test_rec,test_F1,train_F1/test_F1])
                                # Predicciones (lista)
                                Pred_list=[th,ytrain_pred_ind,ytest_pred_ind]
                                # Y almaceno el modelo
                                directory = "Model_Ch"+str(n_channels)+"_W"+str(window_size)+"_Ts"+str(timesteps)+"_Bi"+str(bi)+"_L"+str(n_layers)+"_U"+( (str(n_uds)) if len(str(n_uds))!=1 else ("0"+str(n_uds))) +"_E"+str(n_epochs)+"_TB"+str(n_train_batch)
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
                                "Bidirectional": bi,
                                "Layers": n_layers,
                                "Units": n_uds,
                                "Epochs": n_epochs,
                                "Batch size": n_train_batch,
                                }
                                to_save={'results': results,
                                'params':params,
                                }
                                # Store data (serialize): un archivo para cada bucle de entrenamiento
                                with open(root+ 'Results\Results_Ch%d_W%d_Ts%d_Bi%d_L%d_U%d_E%d_TB%d.pickle' % (n_channels,window_size,timesteps,bi,n_layers,n_uds,n_epochs,n_train_batch), 'wb') as handle:
                                    pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
                ################# Fin del bucle

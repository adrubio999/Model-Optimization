import numpy as np
import pickle
import sklearn as sk
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import os
import sys
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import get_predictions_index,perf_array,split_data,compute_precision_recall_events,pyr,data_path,session,session_path

downsampled_fs=1250


# Definición de pruebas lugar de almacenamiento
TestName="OptimizationTest"
# Carpeta de la prueba
root='C:\Septiembre-Octubre\Model-Optimization\KNN\\'+TestName+'\\'
print(root)

if len(os.listdir(root))==0: #Está vacío, hay que crear
    os.mkdir(os.path.join(root,"Models"))
    os.mkdir(os.path.join(root, "Models\PCA"))
    os.mkdir(os.path.join(root,"Models\KNN"))
    os.mkdir(os.path.join(root, "Results"))    
    os.mkdir(os.path.join(root, "Data"))

with open('C:\ProyectoInicial\Datos_pickle\\x_Amigo2_1.pickle', 'rb') as handle:
    x_amigo=pickle.load(handle)
with open('C:\ProyectoInicial\Datos_pickle\\x_Som_2.pickle', 'rb') as handle:
    x_som=pickle.load(handle)
with open('C:\ProyectoInicial\Datos_pickle\\y_Amigo2_1.pickle', 'rb') as handle:
    y=pickle.load(handle)
with open('C:\ProyectoInicial\Datos_pickle\\y_Som_2.pickle', 'rb') as handle:
    y=np.append(y,pickle.load(handle)) 
y=np.reshape(y,(-1,1))

# Si Dummy==true, prueba reducida solo para funcionaiento
Dummy=False
if Dummy==False:
    tharr=np.linspace(0.05,1,20)
else:
    tharr=np.linspace(0.25,1,4)


# 1: Nº de canales de entrada. Only even numbers are allowed in the CNN2D.
n_channels_arr=[8] 
# 2: Segundos de duración de ventana en que se dividen los datos para hacer separación en train y test
window_size_arr=[60]
# 3: Muestras en cada ventana temporal
# 4 tambien
timesteps_arr=[10,20,32,40]
# 5: Array of percentages of components used in PCA
PCA_arr=[1,0.5,0.25,0.1]
# 6: array of nearest neighbors used in knn
knn_arr=[5,10,15,20]


for n_channels in n_channels_arr:
    if n_channels==8:
        x=np.append(x_amigo,x_som)
    elif n_channels==3:             # 3 canales: primero, piramidal y último
        x=np.append(x_amigo[:, [0,pyr['Amigo2_1'],7]],x_som[:, [0,pyr['Som_2'],7]]) 
    elif n_channels==4:             # 3 canales: primero, piramidal y último
        x=np.append(x_amigo[:, [0,pyr['Amigo2_1']-1,pyr['Amigo2_1'],7]],x_som[:, [0,pyr['Som_2']-1,pyr['Som_2'],7]]) 
    elif n_channels==1:
        x=np.append(x_amigo[:,pyr['Amigo2_1']],x_som[:,pyr['Som_2']])
    x=np.reshape(x,(-1,n_channels))
    for window_size in window_size_arr:
        x_test_or,y_test_or,x_train_or,y_train_or=split_data(x,y,n_channels=n_channels,window_dur=window_size,fs=downsampled_fs,split=0.7)
        # Bucle for para probar time stamps (anchuras de la sliding window distinta)
        for timesteps in timesteps_arr:
            x_train=x_train_or[:len(x_train_or)-len(x_train_or)%timesteps,:].reshape(-1,timesteps*n_channels)
            y_train_aux=y_train_or[:len(y_train_or)-len(y_train_or)%timesteps,:].reshape(-1,timesteps)
            x_test=x_test_or[:len(x_test_or)-len(x_test_or)%timesteps,:].reshape(-1,timesteps*n_channels)
            y_test_aux=y_test_or[:len(y_test_or)-len(y_test_or)%timesteps,:].reshape(-1,timesteps)
            # Dimension adaptation. The input of the model has to be [timesteps, n_channels,1]
            y_train=np.zeros(shape=[x_train.shape[0],1])
            for i in range(y_train_aux.shape[0]):
                y_train[i]=1  if any (y_train_aux[i]==1) else 0
            print("Train Input and Output dimension", x_train.shape,y_train.shape)
            
            y_test=np.zeros(shape=[x_test.shape[0],1])
            for i in range(y_test_aux.shape[0]):
                y_test[i]=1  if any (y_test_aux[i]==1) else 0
            input_shape=x_test.shape[1:]
            # Array de configuraciones posibles.

            for n_pca in PCA_arr:
                print('\nProportion of components:',n_pca)
                # Nº de épocas
                pca = PCA(n_components = int(n_pca*timesteps*n_channels))
                x_train_pca = pca.fit_transform(x_train)
                x_test_pca=pca.transform(x_test)
                print("Shape after PCA"+str(x_train_pca.shape))
                '''El bueno por si hago mas pruebas jeje
                with open(root+'\Models\PCA\pca_Ch%d_W%d_Ts%02d_PCA%1.3f.pickle' % (n_channels,window_size,timesteps,n_pca), 'wb') as handle:
                    pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
                with open(root+'\Models\PCA\pca_Ch%d_W%d_Ts%02d_PCA_%1.3f.pickle' % (n_channels,window_size,timesteps,n_pca), 'wb') as handle:
                    pickle.dump(pca, handle, protocol=pickle.HIGHEST_PROTOCOL)

                for n_knn in knn_arr:
                    print('Nº de canales: %d, Duración de la ventana: %d, Time steps:%d,\n Components percentage: %1.2f, Neigbors: %d' % (n_channels,window_size,timesteps,n_pca,n_knn))
                    knn = KNeighborsClassifier(n_neighbors=n_knn)
                    knn.fit(x_train_pca, y_train)
                    

                    # Predicción. Devuelve una predicción por ventana
                    test_signal = knn.predict_proba(x_test_pca)[:,1]
                    train_signal = knn.predict_proba(x_train_pca)[:,1]
                    print(train_signal.shape)
                    # input("Breakpoing falso XDD")
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
                    with open(root+'\Models\KNN\knn_Ch%d_W%d_Ts%02d_PCA%1.3f_k%02d.pickle' % (n_channels,window_size,timesteps,n_pca,n_knn), 'wb') as handle:
                        pickle.dump(knn, handle, protocol=pickle.HIGHEST_PROTOCOL)

                    # Almaceno en un diccionario
                    results = {
                    "performance": best_performance,
                    "predictions": Pred_list,
                    }
                    params={
                    "N channels": n_channels,
                    "Window size": window_size,
                    "Time steps": timesteps,
                    "N PCA": n_pca,
                    "N neighbors": n_knn,
                    }
                    to_save={'results': results,
                    'params':params,
                    }
                    # Store data (serialize): un archivo para cada bucle de entrenamiento
                    with open(root+'\Results\\resuts_Ch%d_W%d_Ts%02d_PCA%1.3f_k%02d.pickle' % (n_channels,window_size,timesteps,n_pca,n_knn), 'wb') as handle:
                        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
        ################# Fin del bucle'''
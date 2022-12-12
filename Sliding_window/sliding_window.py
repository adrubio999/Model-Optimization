import pickle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from tensorflow import keras
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

from aux_fcn import compute_precision_recall_events,get_predictions_index,prediction_parser,generate_overlapping_windows_fast,integrate_window_to_sample,integrate_window_to_sample_own,get_prediction_indexes, session,pyr
from aux_fcn import load_ripples, session_path


fs=1250
model_window=0.012
# Sliding window, as a proporti on [0:1] of the window size defined by the model
slide_prop=[0.80]

n_sessions=21
n_channels=8
debug=False

tharr=np.linspace(0.05,1,20)
test='CompilationTest'

stride_arr=[model_window*s for s in slide_prop ]
results=np.empty(shape=(n_sessions,len(tharr),5))
with open('C:\Septiembre-Octubre\Model-Optimization\Consensus\Results\\CNN1D_best_model', 'rb') as handle:
  model_dic=(pickle.load(handle))
for stride in stride_arr:
    for s in range(1):
        print(f"Computing model output with a sliding window of {stride} milisecons for the session number {s}")

        with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
            x=pickle.load(handle)
        with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
            y=pickle.load(handle) 
        x_exp=generate_overlapping_windows_fast(x, model_window, stride, fs).squeeze()

        #y_pred_from_parser=prediction_parser(model_dic,x_exp,s)


        
        print(f"\n\n\n\nOriginal data shape: {x.shape}, expanded data shape: {x_exp.shape}")



        #with open("C:\Septiembre-Octubre\Model-Optimization\Sliding_window\Debug_file.txt", "a") as f:
        #    f.write(str(x_exp))
        # Cant use prediction parser 
        x_exp=x_exp.reshape(1,-1,n_channels)
        optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
        model = keras.models.load_model('C:\Septiembre-Octubre\Model-Optimization\Consensus\Models\\CNN1D', compile=False)
        model.compile(loss="binary_crossentropy", optimizer=optimizer)
        y_pred_windowed = model.predict(x_exp, verbose=True).squeeze()

        y_pred = integrate_window_to_sample(y_pred_windowed, model_window, stride, fs, n_samples=x.shape[0], func=np.mean)
        print(f"Output shape extended (1 value per window): {y_pred_windowed.shape}\nCollapsed output shape: {y_pred.shape}")
        ripples=load_ripples(session_path[s])
        y_gt_ind=np.empty(shape=(ripples.shape[0],2))
        y_gt_ind[:,0]=ripples[:,0]/30000
        y_gt_ind[:,1]=ripples[:,2]/30000
        print(y_gt_ind)

        #y_gt_ind=get_predictions_index(y,0.7)

        if debug:
            fig,axs=plt.subplots(4,1,figsize=(14,6))
            muestras=1000
            axs[0].plot(x[:,0])
            axs[1].plot(x_exp[0,:,0])
            axs[2].plot(y_pred[:muestras*50])

            axs[2].legend(['Output from og func','Output from parser','Output from mod function'])
            axs[3].plot(y[:muestras*50])

            plt.show()
        for i,th in enumerate(tharr):
            print('Threshold {:1.3f}'.format(th))
            y_pred_ind=get_predictions_index(y_pred,th)
            y_pred_ind_jul=get_prediction_indexes(y_pred,th,merge_interval=0.012)
            #print(f"Ground truth: {y_gt_ind}, Index: {y_pred_ind_jul}")
            #input("Pausa")
            # Print de indices de cada
            #input(ph)
            if debug:
                fig,axs=plt.subplots(2,1,figsize=(14,6))
                axs[0].plot(y_pred_ind[:,0])
                axs[1].plot(y_gt_ind[:,0])

                axs[0].legend(['Ind from function','Ind from parser','Ind from processing after parser'])
                axs[1].legend(['GT ind own','GT ind andrea'])
                plt.show()
                print(f"Own: {y_pred_ind[0]}, other: {y_gt_ind[0]}")
                
            #print(f"Indices propios: {y_pred_ind}\n Índices con la función de Julio: {y_pred_ind}")
            # s: session number. aux_fcn has dictionaries that assign the correct path
            prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
            print(f'Con los indices aplicados a integrate windowto sample:\nPrecission: {prec}, recall: {rec}, F1: {F1}')
            prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind_jul,y_gt_ind,0)
            print(f'Integrate window to sample -> get indexes fcn from Julio:\nPrecission: {prec}, recall: {rec}, F1: {F1}')
            # Modelo, # th1, #th2, P,R y F1
            results[s][i]=[s,th,prec, rec, F1]
'''
        input(y_gt_ind)
        Validation_results={
            "Slide window":stride,
            "Performance":results,
            }

    with open(f'C:\\Septiembre-Octubre\\Model-Optimization\\Sliding_window\\{test}\Results_Stride{stride:0.3f}.pickle', 'wb') as handle:
            pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    '''
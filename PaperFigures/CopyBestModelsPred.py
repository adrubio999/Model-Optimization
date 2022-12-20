import pickle
import numpy as np
import shutil
import sys
import os
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')
from aux_fcn import session_path
arqs=['SVM']#['XGBOOST','SVM','LSTM','CNN2D','CNN1D']
n_sessions=21
Delete_prev=False

if Delete_prev==True:
    for s in range (n_sessions):
        events_path=session_path[s]+'\events\\Best'
        for filename in os.listdir(events_path):
            if filename[0]=='B' or filename[-4:]=='.mat':
                continue
            input(filename) # Por si la lio nada mas
            os.remove(events_path+'\\'+filename)

    
for a,arq in enumerate(arqs):
    Root=f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\Models\{arq}\Validation'  # Load data (deserialize)
    Model=[]
    Codes=[]
    for filename in os.listdir(Root):
        Codes.append(filename[8:])
        f = os.path.join(Root, filename)
        # checking if it is a file
        with open(f, 'rb') as handle:
            Model.append(pickle.load(handle))
    n_models=len(Codes)
    F1_mean_max=np.empty(shape=n_models)

    for n in range(n_models):
        performance=np.nan_to_num(Model[n]['Performance'])
        n_sessions,n_th,j=np.shape(performance)
        F1_arr=np.empty(shape=(n_sessions,n_th))
        # Solo hay un array de ths
        for i in range(n_sessions):
            F1_arr[i]=performance[i,:,4]
        F1_mean_max[n]=np.max(np.mean(F1_arr,axis=0))

    best_index=(-F1_mean_max).argsort()[0]
    best_model_code=Codes[best_index]
    for s in range (n_sessions):
        events_path=session_path[s]+'\events\\Paper'
        print(events_path)
        for filename in os.listdir(events_path):
            ModelString=f"{arq}_{best_model_code[0:-4]}"

            if ModelString!=filename[:len(ModelString)]:
                continue
            print(events_path+'\\'+filename)
            term=filename.split('_')[-1]
            print(f"{session_path[s]}\events\\Best\\{arq}_{term}")
            shutil.copyfile(f"{events_path}\\{filename}",f"{session_path[s]}\events\\Best\\{arq}_{term}") 
            
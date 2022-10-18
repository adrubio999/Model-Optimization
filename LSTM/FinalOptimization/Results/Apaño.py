import os
import sys
import pickle
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')



for filename in os.listdir('C:\Septiembre-Octubre\Model-Optimization\LSTM\FinalOptimization\Results'):
    print(filename,len(filename))
    
    f = os.path.join('C:\Septiembre-Octubre\Model-Optimization\LSTM\FinalOptimization\Results',filename)
    
    if (filename[0]!='R'):
        continue

    # checking if it is a file
    
    with open(f, 'rb') as handle:
        Saved=(pickle.load(handle))
    '''with open(root+ 'Results\Results_Ch%d_W%d_Ts%d_Bi%d_L%d_U%d_E%d_TB%d.pickle' % (n_channels,window_size,timesteps,bi,n_layers,n_uds,n_epochs,n_train_batch), 'wb') as handle:
                                    pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
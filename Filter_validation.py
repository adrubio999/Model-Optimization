import pickle
from time import time
import numpy as np
import os
import sys
from scipy.signal import butter, lfilter, savgol_filter, filtfilt
from tensorflow import keras
import matplotlib.pyplot as plt
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

def smooth(y, box_pts=0):
	# By default, it depends on the signal
	if box_pts<=0:
		box_pts = int(np.ceil(len(y)/10))
	# Make box
	box = np.ones(box_pts)/box_pts
	# Convolve
	y_smooth = np.convolve(y, box, mode='same')
	return y_smooth


from aux_fcn import compute_precision_recall_events,get_predictions_index,format_predictions,prediction_parser,session,pyr

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, fs, lowcut=70, highcut=250, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def compute_filter_and_envelope(data, fs, lowcut=70, highcut=250, order=3, t_sgolay=0.0334, k_sgolay=4, t_smooth1=0.0030, t_smooth2=0.0065):
	# Filter pyramidal layer to obtain ripple power through time
	pyr_filtered = butter_bandpass_filter(data, fs, lowcut=lowcut, highcut=highcut, order=3)
	# Compute envelope from filtered signal
	win_sgolay = int(2.*np.floor(t_sgolay*fs/2.)+1) # round to nearest odd number
	pyr_envelope = smooth( smooth( savgol_filter(2.*pyr_filtered*pyr_filtered, win_sgolay, k_sgolay), int(t_smooth1*fs)), int(t_smooth2*fs))
	
	return pyr_filtered, pyr_envelope

dur=0.2
fs=1250
beginning_t=7.50
sam=int(fs*dur)
ini_idx=int(beginning_t*fs)



n_sessions=21
n_channels=8
hf_cutoff_arr=[200,250,300,400,500,1000]
lf_cutoff_arr=[1,0.5,0.25,0.1]
tharr=np.linspace(0.05,1,20)
test='Band_pass2'

# Downsampled data
results=np.empty(shape=(n_sessions,len(tharr),5))


with open('C:\Septiembre-Octubre\Model-Optimization\Consensus\Results\\CNN1D_best_model', 'rb') as handle:
  model=(pickle.load(handle))
for hf_cutoff in hf_cutoff_arr:
  for lf_cutoff in lf_cutoff_arr:
    print(f'\n\n\n\nPerforming validation with data filtered in the {hf_cutoff}-{lf_cutoff} band.')
    for s in range(n_sessions): #for aquí
      with open('C:\ProyectoInicial\Datos_pickle\\x_'+session[s]+'.pickle', 'rb') as handle:
          x=pickle.load(handle)
      with open('C:\ProyectoInicial\Datos_pickle\\y_'+session[s]+'.pickle', 'rb') as handle:
          y=pickle.load(handle) 
    
      x_filtered=np.empty_like(x)
      for ch in range(n_channels):
        x_filtered[:,ch]=butter_bandpass_filter(x[:,ch],fs,order=3,highcut=hf_cutoff,lowcut=lf_cutoff)
      input(x_filtered.shape)
      y_pred=prediction_parser(model,x_filtered,s)


      # Ground truth indexes
      y_gt_ind=get_predictions_index(y,0.7)
      
      for i,th in enumerate(tharr):
        print('Threshold {:1.3f}'.format(th))
        y_pred_ind=get_predictions_index(y_pred,th)
        # s: session number. aux_fcn has dictionaries that assign the correct path
        print(y_pred_ind[0],y_gt_ind[0])
        prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
        # Modelo, # th1, #th2, P,R y F1
        input("Pausa")
        results[s][i]=[s,th,prec, rec, F1]
      
      Validation_results={
        "Lowcut":lf_cutoff,
        "Highcut":hf_cutoff,
        "Performance":results,
      }
    if type(lf_cutoff)==int:
      lf_str=f'{lf_cutoff:02d}' 
    else:
      lf_str=f'{lf_cutoff:02.2f}'
    with open(f'C:\\Septiembre-Octubre\\Model-Optimization\\Filters\\{test}\Results_hf{hf_cutoff}_lf{lf_str}.pickle', 'wb') as handle:
          pickle.dump(Validation_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
#fig,axs=plt.subplots(8,1,constrained_layout=True,figsize=(10,7))
#print(x_filtered.shape)
#for i,ax in enumerate(axs):
#    ax.plot(x[ini_idx:ini_idx+sam,i],linewidth=1)
#    ax.plot(x_filtered[ini_idx:ini_idx+sam,i],linewidth=1)
#    ax.plot(x_filtered[ini_idx:sam,i],linewidth=1)
#plt.show()







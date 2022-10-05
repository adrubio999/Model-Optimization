import numpy as np
import os 
import sys
import pandas as pd
from tensorflow import keras
#Diccionario con los canales piramidales. De momento voy a poner el 4º en todos, preguntar a liset
pyr={'Amigo2_1': 0,
	'Som_2': 0,
	0: 2,  # Dlx1_1
	1: 4,  # Thy7_1
	2: 2,      # PV6_1
	3: 1, # PV7xChR2_1
	4: 3,     # Thy9_1
	5: 2,     # Thy1GCam1_1

	6: 4, # Thy7_2
	7: 4, # Thy7_3

	8: 3, # Thy1GCam1_2
	9: 4, # Thy1GCam1_3
	10: 2, # Thy1GCam1_4
	11: 2, # Thy1GCam1_5
	12: 2, # Thy1GCam1_6
	13: 2, # Thy1GCam1_7

	14: 3, # Calb20

	15: 3, # Dlx1_2

	16: 3, # Thy9_2

	17: 0, # PV7xChR2_2
	18: 0, # PV7xChR2_3
	
	19: 5, # Thy10_1
	20: 2, # Thy10_2
        }
session={
	# Training sessions
	-2: "Amigo2_1",
	-1: "Som_2",
	# Original val sessions
	0: "Dlx1_1",
	1: "Thy7_1",
	2: "PV6",
	3: "PV7xChR2_1",
	4: "Thy9_1",
	5: "Thy1GCam1_1",

    6: "Thy7_2",
	7: "Thy7_3",
	8: "Thy1GCam1_2",
	9: "Thy1GCam1_3",
	10: "Thy1GCam1_4",
	11: "Thy1GCam1_5",
	12: "Thy1GCam1_6",
	13: "Thy1GCam1_7",
	14: "Calb20",
	15: "Dlx1_2",
	16: "Thy9_2",
	17: "PV7xChR2_2",
	18: "PV7xChR2_3",
	19: "Thy10_1",
	20: "Thy10_2"
}
# Added +1, the txt wil be read by matlab
shanks={-2: 3, # Amigo2_1 OK
        -1: 3, # Som2 OK
        0: 4,  # Dlx1_1
        1: 3,  # Thy7_1
        2: 3,  # PV6
        3: 4,  # PVxChR2_1
        4: 4,  # Thy9_1
        5: 2,  # Thy1GCam1

        6: 3, # Thy7_2
		7: 3, # Thy7_3

		8: 3, # Thy1GCam1_2
		9: 2, # Thy1GCam1_3
		10: 2, # Thy1GCam1_4
		11: 4, # Thy1GCam1_5
		12: 4, # Thy1GCam1_6
		13: 4, # Thy1GCam1_7

		14: 4, # Calb20

		15: 4, # Dlx1_2

		16: 3, # Thy9_2

		17: 4, # PV7xChR2_2
		18: 4, # PV7xChR2_3
		
		19: 3, # Thy10_1
		20: 3, # Thy10_2
        }
# Para escribir la primera línea en los .txt de predicciones 
data_path={-2: 'C:\ProyectoInicial\Datos\Kilosort\Amigo2_1',
			-1: 'C:\ProyectoInicial\Datos\Kilosort\Som_2',
			0: 'C:\ProyectoInicial\Datos\Kilosort\Dlx1',  # Val
			1: 'C:\ProyectoInicial\Datos\Kilosort\Thy7',  # Val
			2: 'C:\ProyectoInicial\Datos\Kilosort\PV6',
			3: 'C:\ProyectoInicial\Datos\Kilosort\PV7xChR2',
			4: 'C:\ProyectoInicial\Datos\Kilosort\Thy9',
			5:	'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1',

			6: 'C:\ProyectoInicial\Datos\Kilosort\Thy7',
			7: 'C:\ProyectoInicial\Datos\Kilosort\Thy7',
			8: 'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1',
			9: 'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1',
			10:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1',
			11:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1',
			12:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1',
			13:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1',
			14:'C:\ProyectoInicial\Datos\Kilosort\Calb20',
			15:'C:\ProyectoInicial\Datos\Kilosort\Dlx1',
			16:'C:\ProyectoInicial\Datos\Kilosort\Thy9',
			17:'C:\ProyectoInicial\Datos\Kilosort\PV7xChR2',
			18:'C:\ProyectoInicial\Datos\Kilosort\PV7xChR2',
			19:'C:\ProyectoInicial\Datos\Kilosort\Thy10',
			20:'C:\ProyectoInicial\Datos\Kilosort\Thy10',
				
					}
# To store the .txt with the events in the corresponding data path
session_path={-2: 'C:\ProyectoInicial\Datos\Kilosort\Amigo2_1\hippo_2019-07-11_11-57-07_1150um_re_tag',
			-1: 'C:\ProyectoInicial\Datos\Kilosort\Som_2\hippo_2019-07-24_12-01-49_1530um_re_tag',
			0: 'C:\ProyectoInicial\Datos\Kilosort\Dlx1\\2021-02-12_12-46-54',  # Val
			1: 'C:\ProyectoInicial\Datos\Kilosort\Thy7\\2020-11-11_16-05-00',  # Val
			2: 'C:\ProyectoInicial\Datos\Kilosort\PV6\\2021-04-19_14-02-31',
			3: 'C:\ProyectoInicial\Datos\Kilosort\PV7xChR2\\2021-05-18_13-24-33',
			4: 'C:\ProyectoInicial\Datos\Kilosort\Thy9\\2021-03-16_12-10-32',
			5:	'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-18_14-40-16',

			6: 'C:\ProyectoInicial\Datos\Kilosort\Thy7\\2020-11-11_16-21-15',
			7: 'C:\ProyectoInicial\Datos\Kilosort\Thy7\\2020-11-11_16-35-43',
			8: 'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-18_13-16-03',
			9: 'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-18_13-32-27',
			10:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-18_14-56-54',
			11:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-21_14-58-51',
			12:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-21_15-11-32',
			13:'C:\ProyectoInicial\Datos\Kilosort\Thy1GCam1\\2020-12-21_15-26-01',
			14:'C:\ProyectoInicial\Datos\Kilosort\Calb20\\2021-01-22_13-08-20',
			15:'C:\ProyectoInicial\Datos\Kilosort\Dlx1\\2021-02-12_12-24-56',
			16:'C:\ProyectoInicial\Datos\Kilosort\Thy9\\2021-03-16_14-31-51',
			17:'C:\ProyectoInicial\Datos\Kilosort\PV7xChR2\\2021-05-18_13-08-23',
			18:'C:\ProyectoInicial\Datos\Kilosort\PV7xChR2\\2021-05-18_13-48-31',
			19:'C:\ProyectoInicial\Datos\Kilosort\Thy10\\2021-06-01_13-28-27',
			20:'C:\ProyectoInicial\Datos\Kilosort\Thy10\\2021-06-15_15-28-56',
				
					}
# Funciones de metrics

def compute_precision_recall_events(pred_events, true_events, threshold=0, exclude_matched_trues=False, verbose=True):

	'''
	[precision, recall, F1, TP, FN, IOU] = compute_precision_recall(pred_events, true_events)
	Computes all these measures given a cell array with boundaries of pred_events

	Inputs:
		pred_events		 Nx2 matrix with start and end of pred events 
		true_events		 Mx2 matrix with start and end of true events
		threshold		   Threshold to IoU. By default is 0
		exclude_matched_trues False by defaut (one true can match many predictions)

	Output:
		precision		   Metric indicating the percentage of correct 
							predictions out of total predictions
		recall			  Metric indicating the percentage of true events 
							predicted correctly
		F1				  Metric with a measure that combines precision and recall.
		TP				  Nx1 matrix indicating which pred event detected a true event, and
							a true positive (True) or was false negative (False)
		FN				  Mx1 matrix indicating which true event has been detected (False)
							or not (True)
		IOU				 NxM array with intersections over unions

	A Navas-Olive, LCN 2020
	'''

	# Check similarity of pred and true events by computing if intersection over union > 0
	# Compute IOU
	[IOU, IOU_pred, IOU_true] = intersection_over_union(pred_events, true_events)
	# IOU_pred-> IOU de cada evento predecido
	# IOU_true-> IOU de la GT


 # Excluye los true coincidentes
	if exclude_matched_trues:
		# Take maximal IOUs, and make the rest be zero
		pred_with_maxIOU = np.argmax(IOU, axis=0)
		IOU_pred_one_true_match = np.zeros_like(IOU)
		for itrue, ipred in enumerate(pred_with_maxIOU):
			IOU_pred_one_true_match[ipred, itrue] = IOU[ipred, itrue]
		# True positive: Predicted event that has a IoU with any true > 0
		TP = (IOU_pred_one_true_match.sum(axis=1) > threshold)
		# False negative: Predicted event that has not a IoU with any true
		FN = (IOU_true <= threshold)

	else:
		# True positive: Predicted event that has a IoU with any true > 0
		# 
		TP = (IOU_pred>threshold) 
		# False negative: Predicted event that has not a IoU with any true
		FN = (IOU_true<=threshold)  
	
	# Precision and recall
	precision = np.mean(TP)      # Media de verdaderos positivos: 1 si todas las predicciones son aciertos
	recall = 1. - np.mean(FN)    # 1-media de falsos negativos: 1 si toda la GT está presente en los aciertos
	F1 = 2. * (precision * recall) / (precision + recall)  
	if (precision + recall) == 0:
		F1 = 0.
	else:
		F1 = 2. * (precision * recall) / (precision + recall)

	if verbose:
		print('precision =', precision)
		print('recall =', recall)
		print('F1 =', F1)
	
	# Variable outputs
	return precision, recall, F1, TP, FN, IOU

def intersection_over_union(x, y):
	'''
	IOU = intersection_over_union(x, y) computes the percentage of 
	intersection over their union between every two pair of intervals 
	x and y.
	
	Inputs:
	  x	 Nx2 array with beginnings and ends of 1D events
	  y	 Mx2 array with beginnings and ends of 1D events
	
	Output:
	  IOU   NxM array with intersections over unions
	  IOUx  (optional) Nx1 array with indexes of y events with maximal IOU 
			It's zero if IOU=0 for all y events
	  IOUy  (optional) Mx1 array with indexes of x events with maximal IOU 
			It's zero if IOU=0 for all x events
	
	A Navas-Olive, LCN 2020
	'''

	if (len(x)>0) and (len(y)>0):

		# Initialize
		Intersection = np.zeros((x.shape[0],  y.shape[0]))
		Union = np.ones((x.shape[0],  y.shape[0]))
		# Go through every y (beginning-end) pair
		for iy in range(y.shape[0]):
			# Intersection duration: difference between minimum end and maximum ini
			Intersection[:, iy] = np.maximum( np.minimum(x[:, 1], y[iy, 1]) - np.maximum(x[:, 0], y[iy, 0]), 0)
			# Union duration: sum of durations of both events minus its intersection
			Union[:, iy, None] = np.diff(x, axis=1) + np.diff(y[iy, :]) - Intersection[:, iy, None]

		# Compute intersection over union
		IOU = Intersection / Union

		# Compute which events from y have maximal IOU with x
		IOUx = np.max(IOU, axis=1, keepdims=True)

		# Compute which events from x have maximal IOU with y
		IOUy = np.max(IOU, axis=0, keepdims=True)

		# Optional outputs
		return IOU, IOUx, IOUy
		
	elif len(x)==0:
		
		print('x is empty. Cant perform IoU')
		return np.array([]), np.array([]), np.zeros((y.shape[0], 1))
		
	elif len(y)==0:
		
		print('y is empty. Cant perform IoU')
		return np.array([]), np.zeros((1, x.shape[0])), np.array([])

def relate_true_pred(true_events, pred_events, pred_events_offset, middles, minimums, separation=0, fuse=True):

	print(len(pred_events))
	# Fuse events that start just after last event
	if fuse:
		ini_times = pred_events[:, 0]
		end_times = pred_events[:, 1]

		ini_times_offset = pred_events_offset[:, 0]
		end_times_offset = pred_events_offset[:, 1]

		# Append first initial time
		new_inis = [ini_times[0]]
		new_ends = []

		new_inis_offset = [ini_times_offset[0]]
		new_ends_offset = []
		# start,stop
		for ievent in range(2,len(ini_times)):
			# If previous event is separated from beginning of this event
			# then, we append new end and beginning.
			# Si el evento previo está lo suficientemente separado del actual, se añaden los tiempos finales e iniciales
			if end_times[ievent-1] <= (ini_times[ievent] - 0.0064):
				new_ends.append(end_times[ievent-1])
				new_inis.append(ini_times[ievent])

				new_ends_offset.append(end_times_offset[ievent-1])
				new_inis_offset.append(ini_times_offset[ievent])
		
		new_ends.append(end_times[-1])
		new_ends_offset.append(end_times_offset[-1])


		# Replace
		ini_times = np.array(new_inis).reshape(-1,1)
		end_times = np.array(new_ends).reshape(-1,1)

		pred_events = np.concatenate((ini_times, end_times), axis=1)

		ini_times_offset = np.array(new_inis_offset).reshape(-1,1)
		end_times_offset = np.array(new_ends_offset).reshape(-1,1)

		pred_events_offset = np.concatenate((ini_times_offset, end_times_offset), axis=1)

	print(len(pred_events))


	EPSILON = 0.00001
	
	num_trues = len(true_events)
	num_preds = len(pred_events)

	pred_events_unique = []
	true_positives_unique = np.zeros(num_preds, dtype=int)

	true_positives = np.zeros(num_preds, dtype=int)
	false_negatives = np.ones(num_trues, dtype=int)

	correlation = np.zeros((num_trues, num_preds), dtype=float)

	
	for i_true, true in enumerate(true_events):
		for i_pred, pred in enumerate(pred_events_offset):

			# End of pred is after end of true
			if pred[1] >= true[1]:

				# Check if start of pred is before or close after end of true
				if (pred[0] - separation) <= true[1]:
					# match
					if (false_negatives[i_true] != 0):
						# New matched true event
						true_positives_unique[i_pred] = 1


					true_positives[i_pred] = 1
					false_negatives[i_true] = 0

					# Compute IoU (with epsilon for cases when separation > 0)
					inter = np.minimum(true[1], pred[1]) - np.maximum(true[0], pred[0])
					union = ((true[1] - true[0]) + (pred[1] - pred[0])) - inter
					iou = np.maximum((inter / union), 0.) # It can be negative when separation > 0
					correlation[i_true, i_pred] = iou + EPSILON

			# Start of pred is before start of true
			elif pred[0] <= true[0]:

				# Check if end of pred is after or close before start of true
				if (pred[1] + separation) >= true[0]:
					# match
					if (false_negatives[i_true] != 0):
						# New matched true event
						true_positives_unique[i_pred] = 1

					true_positives[i_pred] = 1
					false_negatives[i_true] = 0
					
					# Compute IoU (with epsilon for cases when separation > 0)
					inter = np.minimum(true[1], pred[1]) - np.maximum(true[0], pred[0])
					union = ((true[1] - true[0]) + (pred[1] - pred[0])) - inter
					iou = np.maximum((inter / union), 0.) # It can be negative when separation > 0 
					correlation[i_true, i_pred] = iou + EPSILON

			# Pred starts after true and ends before
			else:
				# match
				if (false_negatives[i_true] != 0):
					# New matched true event
					true_positives_unique[i_pred] = 1

				true_positives[i_pred] = 1
				false_negatives[i_true] = 0
				
				# Compute IoU (with epsilon for cases when separation > 0)
				inter = np.minimum(true[1], pred[1]) - np.maximum(true[0], pred[0])
				union = ((true[1] - true[0]) + (pred[1] - pred[0])) - inter
				iou = np.maximum((inter / union), 0.) # It can be negative when separation > 0
				correlation[i_true, i_pred] = iou + EPSILON

	# Compute precision, recall and F1
	precision = np.sum(true_positives) / num_preds
	recall = 1. - (np.sum(false_negatives) / num_trues)
	F1 = 2. * (precision * recall) / (precision + recall)

	# If pred does not match any true its lag is NaN
	lags_ms = np.empty(num_preds)
	lags_ms[:] = np.nan
	lags_per = np.empty(num_preds)
	lags_per[:] = np.nan
	lags_middles = np.empty(num_preds)
	lags_middles[:] = np.nan
	lags_minimums = np.empty(num_preds)
	lags_minimums[:] = np.nan

	# Esta sección computa el retardo en la detección?
	for i_pred, pred in enumerate(pred_events):
		# If pred is a true positive then compute lag
		if true_positives[i_pred] == 1:

			# Get first true that matched pred
			i_true = np.argmax(correlation[:, i_pred])
			lags_ms[i_pred] = pred_events[i_pred][0] - true_events[i_true][0]
			lags_per[i_pred] = lags_ms[i_pred] / (true_events[i_true][1] - true_events[i_true][0])
			lags_middles[i_pred] = pred_events[i_pred][0] - middles[i_true]
			lags_minimums[i_pred] = pred_events[i_pred][0] - minimums[i_true]




	# Compute mean lag ignoring NaNs and outliers
	mean_lag_ms = np.mean(lags_ms[abs(lags_ms - np.nanmean(lags_ms)) < 5 * np.nanstd(lags_ms)])
	std_lag_ms = np.std(lags_ms[abs(lags_ms - np.nanmean(lags_ms)) < 5 * np.nanstd(lags_ms)])
	mean_lag_per = np.mean(lags_per[abs(lags_per - np.nanmean(lags_per)) < 5 * np.nanstd(lags_per)])
	std_lag_per = np.std(lags_per[abs(lags_per - np.nanmean(lags_per)) < 5 * np.nanstd(lags_per)])
	mean_lag_middles = np.mean(lags_middles[abs(lags_middles - np.nanmean(lags_middles)) < 5 * np.nanstd(lags_middles)])
	std_lag_middles = np.std(lags_middles[abs(lags_middles - np.nanmean(lags_middles)) < 5 * np.nanstd(lags_middles)])
	mean_lag_minimums = np.mean(lags_minimums[abs(lags_minimums - np.nanmean(lags_minimums)) < 5 * np.nanstd(lags_minimums)])
	std_lag_minimums = np.std(lags_minimums[abs(lags_minimums - np.nanmean(lags_minimums)) < 5 * np.nanstd(lags_minimums)])

	print("**** New ****")
	print("Precision: %f"%(precision))
	print("Recall: %f"%(recall))
	print("F1: %f"%(F1))
	print("Lag: %f ± %f ms (%.3f%% ± %.3f%%)"%(mean_lag_ms, std_lag_ms, mean_lag_per, std_lag_per))
	print("Lag middle: %f ± %f ms (minimums %.f ± %.f)"%(mean_lag_middles, std_lag_middles, mean_lag_minimums, std_lag_minimums))

def load_ripples (path, verbose=False):
	try:
		dataset = pd.read_csv(path+"/ripples.csv", delimiter=' ', header=0, usecols = ["ripIni", "ripMiddle", "ripEnd", "type", "shank"])
	except:
		print(path+"/ripples.csv file does not exist.")
		sys.exit()

	ripples = dataset.values
	ripples = ripples[np.argsort(ripples, axis=0)[:, 0], :]
	if verbose:
		print("Loaded ripples: ", len(ripples))

	return ripples
# Funciones propias

# For the moment I'll leave the original get_predictions_index comented, just in case the faster one fails
'''def get_predictions_index(predictions,threshold=0.5):
    aux=np.copy(predictions)
    aux[aux>=threshold]=1
    aux[aux<threshold]=0
    pred_indexes =[]
    dif=np.diff(aux,axis=0)
    for i_pred,pred in enumerate(dif):
        if pred==1:
            pred_indexes.append(i_pred)
        elif pred==-1:
            pred_indexes.append(i_pred)
    if len(pred_indexes)%2==1:
        pred_indexes=pred_indexes[:-1]    
    pred_indexes=np.array(pred_indexes).reshape(-1,2)
    return(np.array(pred_indexes).reshape(-1,2))'''




def get_predictions_index(predictions,threshold=0.5):
	aux=np.copy(predictions)
	aux[aux>=threshold]=1
	aux[aux<threshold]=0
	pred_indexes =[]
	dif=np.diff(aux,axis=0)
	begin_indexes=np.where(dif==1)[0]
	end_indexes=np.where(dif==-1)[0]
	#print(begin_indexes.shape,end_indexes.shape)
	if len(begin_indexes)>len(end_indexes):
		begin_indexes=begin_indexes[:-1]
	elif len(begin_indexes)<len(end_indexes):
		end_indexes=end_indexes[1:]
	#print(len(begin_indexes),len(end_indexes))
	pred_indexes=np.empty(shape=(len(begin_indexes),2))
	pred_indexes[:,0]=begin_indexes
	pred_indexes[:,1]=end_indexes
	#print(pred_indexes)
	return pred_indexes

def perf_array(y,y_gt,tharr):
    # Dummy: para hacer pruebas reducidas de funcionamiento 
    performances=[]
    for th in tharr:
        y_gt_ind=get_predictions_index(y_gt,th)
        print('Umbral :', th)
        y_pred_ind=get_predictions_index(y,th)
        prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
        performances=np.append(performances,[th,prec,rec,F1])
    performances=np.reshape(np.nan_to_num(performances),(-1,4))
    return performances

def split_data(x,y,n_channels,window_dur=60,fs=1250,split=0.7):
	
    n_samples_window=window_dur*fs
    n_windows=len(x)//n_samples_window
    x_test=[]
    x_train=[]
    y_train=[]
    y_test=[]
    rand_arr= np.random.rand(n_windows)         # Distribución normal, quizá no sea lo más correcto
    print(rand_arr)
    for i in range(n_windows):
        if rand_arr[i]>=split:
            x_test=np.append(x_test,x[i*n_samples_window:(i+1)*n_samples_window])
            y_test=np.append(y_test,y[i*n_samples_window:(i+1)*n_samples_window])
        else:
            x_train=np.append(x_train,x[i*n_samples_window:(i+1)*n_samples_window])
            y_train=np.append(y_train,y[i*n_samples_window:(i+1)*n_samples_window])
            
    x_test=np.reshape(x_test,(-1,n_channels))
    x_train=np.reshape(x_train,(-1,n_channels))
    y_test=np.reshape(y_test,(-1,1))
    y_train=np.reshape(y_train,(-1,1))
    return x_test,y_test,x_train,y_train
# Solo hace falta pasar downsample_fs si no es la habitual de 1250

def format_predictions(preds,session_number,filename,downsample_fs=1250):
	TestName=filename.split('\\')[1]
	session_name=session[session_number]
	path=session_path[session_number]+'\events\\'+ TestName
	if not os.path.exists(path):
		os.mkdir(path)
	path=path+'\\'+filename.split('\\')[2]
	f=open(path,'w')
	f.write(data_path[session_number]) 
	f.write('\n')
	f.write('shank_to_plot='+str(shanks[session_number])) 
	f.write('\n')

	preds=preds/downsample_fs
	for pred in preds:
		#print(str(pred)[1:-2])
		
		f.write(str(pred[0])+' ')
		f.write(str(pred[1]))
		f.write('\n')
	f.close()
	return

#  Return a string of fixed length from a int, adding zeros to de left if necessary
def str_of_fixed_length(n,length):
	# n: int to be casted to string 
	# length: length of the desired string
	s=''
	if len(str(n))!=length:
		for i in range (length-len(str(n))):
			s=s+"0"
		s=s+str(n)
	else:
		s=str(n)
	return s

# Return a CNN1D model, depending on the shape of the input
def build_cnn1d_model(timesteps,n_channels):
	keras.backend.clear_session()

	# input layer
	model = keras.models.Sequential()
	# Each convolutional layer is followed by a Batch Normalization layer, and then an activation layer
	# Capa 1
	model.add(keras.layers.Conv1D(filters=4, kernel_size=timesteps//8, strides=timesteps//8, padding='valid', input_shape=(timesteps,n_channels)))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.1))

	# Capa 2
	model.add(keras.layers.Conv1D(filters=2, kernel_size=1, strides=1, padding='valid'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.1))

	# Capa 3
	model.add(keras.layers.Conv1D(filters=8, kernel_size=2, strides=2, padding='valid'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.1))

	# Capa 4
	model.add(keras.layers.Conv1D(filters=4, kernel_size=1, strides=1, padding='valid'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.1))

	# Capa 5
	model.add(keras.layers.Conv1D(filters=16, kernel_size=2, strides=2, padding='valid'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.1))

	# Capa 6
	model.add(keras.layers.Conv1D(filters=8, kernel_size=1, strides=1, padding='valid'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.1))

	# Capa 7
	model.add(keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, padding='valid'))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.LeakyReLU(alpha=0.1))

	model.add(keras.layers.Dense(1, activation="sigmoid"))



	opt = keras.optimizers.Adam(learning_rate=0.001)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['mse'])
	return model
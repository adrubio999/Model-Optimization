import numpy as np
import os 
import sys
import pandas as pd
import pickle
from tensorflow import keras
from xgboost import XGBClassifier
sys.path.insert(1,'C:\Septiembre-Octubre\Model-Optimization')

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

	15: 2, # Dlx1_2

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

		16: 4, # Thy9_2

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

def fcn_save_pickle(name,x):
	'''
	void fcn_save_pickle(name,x) \n
	Generates a pickle file in named n, containing x\n
	'''
	with open(name, 'wb') as handle:
			pickle.dump(x, handle, protocol=pickle.HIGHEST_PROTOCOL)
	return 

def fcn_load_pickle(name):
	'''
	[x] = fcn_load_pickle(name) loads the content of the pickle file to x
	'''
	with open(name, 'rb') as handle:
    		return( pickle.load(handle) )


def get_predictions_index(predictions,threshold=0.5):
	'''
		[pred_indexes] = get_predictions_index(predictions, thershold)\n
		Returns the begining and ending samples of the events above a given threshold\n
		Inputs:
			predictions:	X, array with the continuous output of a model (even the Gt)
			threshold:		float, signal intervals above this value will be considered events
		
		Output:		
			pred_indexes:	Nx2, array containing the begining and ending index sample of the events
		'''
	aux=np.copy(predictions)
	aux[aux>=threshold]=1
	aux[aux<threshold]=0
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
    '''
		[performances] = perf_array(y, y_gt, tharr)\n
		Returns the array of performances of the model response with a given array of tresholds\n
		Inputs:
			y:				n, array with the model output
			y_gt:		n, array with the GT signal (0-1) generated from the tagged events 
			tharr:      th, array with the different threshold values that want to be analyzed

		Output:		
			performances:	[th x 4], each row of the matrix contains the performance of the model for the corresponding threshold, in the form: [theshold,prec,rec,F1]
		'''
    performances=[]
    y_gt_ind=get_predictions_index(y_gt,0.5)
    for th in tharr: 
        print(f'Umbral :{th:0.3f}')
        y_pred_ind=get_predictions_index(y,th)
        prec,rec,F1,a,b,c=compute_precision_recall_events(y_pred_ind,y_gt_ind,0)
        performances=np.append(performances,[th,prec,rec,F1])
    performances=np.reshape(np.nan_to_num(performances),(-1,4))
    return performances

def split_data(x,y,window_dur=60,fs=1250,split=0.7):
    '''
    [x_val,y_val,x_train,y_train] = split_data(x,y,window_dur,fs,split)\n
    Performs the data train-validation split, with the proportion specified in 'split' going to train. The data is shuffled in windows of 'window_dur' seconds\n
    Inputs:
    	    x:			[n X n_channels] matrix with the LFP values of the session
    	    y:			[n,]	ground truth signal generated from the tagged events (0,1)
		window_dur: float, length in seconds of the chunks that will be asigned randomly to train or validation subsets
		fs:			int, sampling frequency of x
		split:		float, proportion of windows that will be asigned to the train subset (the final proportion will diverge, being random)
    
	Output:		
    	x_val:		[v x n_channels]: validation subset input 
		y_val:			[v,]: validation subset output 
		x_train:		[t x n_channels]: training subset input 
		y_train:		[t,]: training subset output
    '''
    n_samples_window=window_dur*fs
    n_windows=len(x)//n_samples_window
    x_val=[]
    x_train=[]
    y_train=[]
    y_val=[]
    n_channels=x.shape[1]
    rand_arr= np.random.rand(n_windows)    
    print(rand_arr)
    for i in range(n_windows):
        if rand_arr[i]>=split:
            x_val=np.append(x_val,x[i*n_samples_window:(i+1)*n_samples_window])
            y_val=np.append(y_val,y[i*n_samples_window:(i+1)*n_samples_window])
        else:
            x_train=np.append(x_train,x[i*n_samples_window:(i+1)*n_samples_window])
            y_train=np.append(y_train,y[i*n_samples_window:(i+1)*n_samples_window])
            
    x_val=np.reshape(x_val,(-1,n_channels))
    x_train=np.reshape(x_train,(-1,n_channels))
    y_val=np.reshape(y_val,(-1,1))
    y_train=np.reshape(y_train,(-1,1))
    return x_val,y_val,x_train,y_train
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


# Returns the output signal of the model in params, withe the correct signal shape

def prediction_parser(params,arq,x,s):
	'''
	[y] = prediction_parser(params, x,s) 
	Computes the output of the model passed in params \n
	Inputs:		
		params:		dictionary with the model parameters and training performance results
		x:			[n x 8] lfp data,subsampled and z-scored 
		s:			int, session number

	Output: 
		y: (n) shape array with the output of the evaluated model
	'''
	# Arquitecture and type of parameter load
	n_channels=params['Params']['N channels']
	timesteps=params['Params']['Time steps']

	
	print(arq,n_channels,timesteps)
	# Input shape: number of channels
	if n_channels==3:
		x=x[:,[0,pyr[s],7]]
	elif n_channels==1:
		x=x[:,pyr[s]].reshape(-1,1)
	print(x.shape)
	input_len=x.shape[0]
	# Input shape: timesteps
	if arq=='XGBOOST':
		x=x[:len(x)-len(x)%timesteps,:].reshape(-1,timesteps*n_channels)
		y_predict= np.zeros(shape=(input_len,1,1))
		# model load
		xgb=XGBClassifier()
		xgb.load_model(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModels\\{arq}.model')
		windowed_signal=xgb.predict_proba(x)[:,1]
		for i,window in enumerate(windowed_signal):
			y_predict[i*timesteps:(i+1)*timesteps]=window
	elif arq=='SVM':
		x=x[:len(x)-len(x)%timesteps,:].reshape(-1,timesteps*n_channels)
		y_predict= np.zeros(shape=(input_len,1,1))
		# model load
		with open(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModels\\{arq}.model', 'rb') as handle:
			clf=pickle.load(handle)
		windowed_signal= clf.predict_proba(x)[:,1]
		for i,window in enumerate(windowed_signal):
			y_predict[i*timesteps:(i+1)*timesteps]=window
        # y_predict: after expanding the windows, to be compatible with perf array
	elif arq=='LSTM'or arq=='LSTMcte':
		x=x[:len(x)-len(x)%timesteps,:].reshape(-1,timesteps,n_channels)
		print(x.shape)
		print(input_len%timesteps)
		# Model load
		model = keras.models.load_model(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModels\\{arq}')
		y_predict = model.predict(x,verbose=1)
		y_predict=y_predict.reshape(-1,1,1)
		y_predict=np.append(y_predict,np.zeros(shape=(input_len%timesteps,1,1))) if (input_len%timesteps!=0) else y_predict
	elif arq=='CNN1D':
		x=x.reshape(1,-1,n_channels)
		optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)
		model = keras.models.load_model(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModels\\{arq}', compile=False)
		model.compile(loss="binary_crossentropy", optimizer=optimizer)

		windowed_signal = model.predict(x, verbose=True)
		windowed_signal=windowed_signal.reshape(-1)
		y_predict=np.zeros(shape=(input_len,1,1))
		for i,window in enumerate(windowed_signal):
			y_predict[i*timesteps:(i+1)*timesteps]=window
	elif arq=='CNN2D':
		model = keras.models.load_model(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\BestModels\\{arq}\\model_prob_vf.h5')
		x=x[:len(x)-len(x)%timesteps,:].reshape(-1,timesteps,n_channels,1)
		y_predict= np.zeros(shape=(input_len,1,1))
		windowed_signal= model.predict(x,verbose=1)
		print(windowed_signal.shape)
		for i,window in enumerate(windowed_signal):
			y_predict[i*timesteps:(i+1)*timesteps]=window
        
	return(y_predict.reshape(-1))

#########################################################################################
#					Functions for sliding window										
#########################################################################################
# Expands the data as sliding windows
def generate_overlapping_windows_fast(data, window_size, stride, fs):
	'''
	Expand data by concatenating windows according to window_size and stride

	Inputs:
	-------
		data: numpy array (n_samples x n_channels) 
			LFP data

		stride: float (s)
			Length of stride in seconds (step taken by the window). Note that window size is given
			by the model (currently 32ms)

		fs: integer (Hz)
			sampling frequency in Hz of LFP data 


	Outputs: 
	--------
		new_data: numpy array (1, n_samples', n_channels)
			Numpy array containing the expanded data.

	23-May-2022: Julio E
	'''

	assert window_size>=stride, 'stride must be smaller or equal than window size (32ms) to avoid discontinuities'
	window_pts = int(window_size * fs)
	stride_pts = int(stride * fs)
	assert stride_pts>0, 'pred_every must be larger or equal than 1/downsampled_fs (>0.8 ms)'
	num_windows = np.ceil((data.shape[0]-window_pts)/stride_pts).astype(int)+1 
	remaining_pts = (num_windows-1)*stride_pts + window_pts - data.shape[0]
	new_data = np.zeros(((num_windows+1)*window_pts,data.shape[1])) #add one empty window for the cnn

	for win_idx in range(num_windows-1):
		win = data[win_idx*stride_pts:win_idx*stride_pts+window_pts,:]
		new_data[win_idx*window_pts:(win_idx+1)*window_pts,:]  = win

	new_data[(win_idx+1)*window_pts:-remaining_pts-window_pts,:] = data[(win_idx+1)*stride_pts:, :]
	new_data = np.expand_dims(new_data, 0)

	return new_data

# Contracts the predictions to the original data shape of the session
def integrate_window_to_sample(win_data, window_size, stride, fs, n_samples=None, func=np.mean):
	'''
	Expand data from windows to original samples taking into account stride size

	Inputs:
	-------
		win_data: numpy array (n_windows,) 
			data for each window to be expanded into samples

		stride: float (s)
			Length of stride in seconds (step taken by the window). Note that window size is given
			by the model (currently 32ms)

		fs: integer (Hz)
			sampling frequency in Hz

		n_samples: integer
			desired number of samples. For instance, last window may be half empty (due to zero paddings).

		func: arithmetic function
			function to be applied when there is more than one window referencing the same sample (
			overlapping due to stride/window_size missmatch).

	Outputs: 
	--------
		new_data: numpy array (1, n_samples', n_channels)
			Numpy array containing the expanded data.

	23-May-2022: Julio E
	'''

	assert window_size>=stride, 'stride must be smaller or equal than window size (32ms) to avoid discontinuities'
	window_pts = int(window_size * fs)
	stride_pts = int(stride * fs)
	assert stride_pts>0, 'pred_every must be larger or equal than 1/downsampled_fs (>0.8 ms)'

	max_win_overlap = np.ceil(window_pts/stride_pts).astype(int) 
	max_num_win = win_data.shape[0]

	if isinstance(n_samples, type(None)):
		n_samples = (max_num_win-1)*stride_pts + window_pts

	sample_data = np.empty((n_samples,))
	win_list = []
	for sample in range(0, n_samples, stride_pts):
		if len(win_list) == 0: #first stride simply append window 0
			win_list.append(0)
		else:
			win_list.append(win_list[-1]+1) #append new window
			if len(win_list)>max_win_overlap: #pop left-most window if aready maximum overlapping
				win_list.pop(0)
			if win_list[-1]>=max_num_win: #discard added window if beyond maximum number of windows
				win_list.pop(-1)
		#print(win_data[win_list])
		#input(func(win_data[win_list]))
		sample_data[sample:sample+stride_pts] = func(win_data[win_list])

	return sample_data

def integrate_window_to_sample_own(y_exp, window_size, stride, fs, n_samples):
	window_pts = int(window_size * fs)
	stride_pts = int(stride * fs)
	y_exp=np.hstack([y_exp,np.zeros(shape=window_pts)])

	out=np.zeros(shape=(n_samples))
	for sample in range(0,n_samples,stride_pts):
		out[sample:sample+stride_pts]=np.mean(y_exp[sample:sample+window_pts])
	return(out)
	


def get_prediction_indexes(y_pred_sample, threshold, downsampled_fs=1250, merge_interval=0.032, min_duration=0.020):

	# Beginnings and ends
	inis = np.argwhere(np.diff(1*(y_pred_sample > threshold))>0)
	ends = np.argwhere(np.diff(1*(y_pred_sample > threshold))<0)

	if len(inis) > 0:
		
		# Check they have the same length
		if inis[0,0] > ends[0,0]: inis = np.vstack((np.array([0]), inis))
		if inis[-1,0] > ends[-1,0]: ends = np.vstack((ends, np.array([len(y_pred_sample)-1])))

		# In seconds
		y_pred = np.hstack((inis, ends))/downsampled_fs

		# Merge if they are very close
		y_pred_merged = y_pred[:1,:]
		for ini, end in y_pred[1:]:
			# Too close
			if (y_pred_merged[-1,1]+merge_interval) >= ini:
				y_pred_merged[-1,1] = end
			# Not merge
			else:
				y_pred_merged = np.vstack((y_pred_merged, np.array([ini,end])))

		# Minimum duration
		durations = np.diff(y_pred_merged, axis=1)
		y_pred_dur = y_pred_merged[durations.flatten()>=min_duration]

	else:
		y_pred_dur = np.empty((0,2))
	print(y_pred_dur.shape)
	return y_pred_dur
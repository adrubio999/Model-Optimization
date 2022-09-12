import numpy as np

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

# Funciones propias

def get_predictions_index(predictions,threshold=0.5):
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
    return(np.array(pred_indexes).reshape(-1,2))

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
def format_predictions(preds,path,downsample_fs=1250):
	f=open(path,'w')
	preds=preds/downsample_fs
	for pred in preds:
		#print(str(pred)[1:-2]) 
		f.write(str(pred)[1:-2]) # Elimino los paréntesis
		f.write('\n')
	f.close()
	return
#Diccionario con los canales piramidales. De momento voy a poner el 4º en todos, preguntar a liset
pyr={0: 3,  # Dlx1
	1: 3,  # Thy7
	'Amigo2_1': 3,
	'Som_2': 3,
	2: 3,      # PV6
	3: 3, # PV7xChR2
	4: 3,     # Thy9
	5: 3,     # Thy1GCam1
	}
session={
	0: "Dlx1",
	1: "Thy7",
	2: "PV6",
	3: "PV7xChR2",
	4: "Thy9",
	5: "Thy1GCam1",
}

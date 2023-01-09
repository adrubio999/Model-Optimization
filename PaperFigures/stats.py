import numpy as np
import scipy 
import scipy.io
import pandas as pd
import pickle 
print("Paco")
with open(f'C:\Septiembre-Octubre\Model-Optimization\PaperFigures\F1_arr_best_means.pickle', 'rb') as handle:
        a=pickle.load(handle)
print(a)
F1={'XGBOOST': a[0],
                'SVM': a[1],
                'LSTM': a[2],
                'CNN2D': a[3],
                'CNN1D':  a[4]}
scipy.io.savemat('F1.mat',F1)
#print(F1)
#print(pg.kruskal(data=F1,dv=))
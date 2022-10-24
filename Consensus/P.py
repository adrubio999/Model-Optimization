import pickle
with open('C:\Septiembre-Octubre\Model-Optimization\Consensus\Results\\CNN2D_best_model',  'rb') as handle:
	load=pickle.load(handle)
print(load)
Params={'N channels': 8,
        'Time steps': 40,
}
results ={ 'Params':Params,
           'Performance': load['results']['Performance'],
}
save={'type':load['type'],
        'test_name': load['test_name'],
        'code' : load['code'],
        'best_th' : load['best_th'],
        'results' : results,
}
print(save)
with open('C:\Septiembre-Octubre\Model-Optimization\Consensus\Results\\CNN2D_best_model_new', 'wb') as handle:
    pickle.dump(save, handle, protocol=pickle.HIGHEST_PROTOCOL)
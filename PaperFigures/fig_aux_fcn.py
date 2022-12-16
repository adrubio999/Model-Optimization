import numpy as np
import matplotlib as mpl
colors_dic={'XGBOOST': (25/255,146/255,81/255),
          'SVM': (174/255,2/255,128/255),
          'LSTM': (255/255,91/255,34/255),
          'CNN2D': (255/255,190/255,13/255),
          'CNN1D': (47/255,182/255,206/255)}
def define_colors_prec(var,n):
    colors=[]
    alpha=[]
    cmap_grey=mpl.colormaps['Greys']
    cmap_blue=mpl.colormaps['PuBu']    
    var=np.array(var,dtype=float)
    # Custom colors depending on var
    inds=(-var).argsort()[:n]
    max_ind=np.max(var[inds])
    min_ind=np.min(var[inds])
    
    for i,F1 in enumerate(var):
        if i in inds:
            colors.append(cmap_blue((F1-min_ind)/(max_ind-min_ind)*0.5+0.5))
            alpha.append(1)
        else:
            colors.append(cmap_grey(0.4))
            alpha.append(0.1)
    return colors,alpha,inds

def define_colors(var,n,arq):
    colors=[]
    alpha=[]
    cmap_grey=mpl.colormaps['Greys']  
    var=np.array(var,dtype=float)
    # Custom colors depending on var
    inds=(-var).argsort()[:n]
    max_ind=np.max(var[inds])
    min_ind=np.min(var[inds])
    var=var[inds]
    for i,F1 in enumerate(var):
        if i in inds:
            #if arq==''
            scaling=(F1-min_ind)/(max_ind-min_ind)*0.5+0.5
            #colors.append(cmap_blue((F1-min_ind)/(max_ind-min_ind)*0.5+0.5))
            a=tuple([x*scaling for x in list(colors_dic[arq])])
            colors.append(a)
            alpha.append(1)
        else:
            colors.append(cmap_grey(0.4))
            alpha.append(0.1)
    # The colors are returned in decreasing order with respect to var
    return colors,alpha,inds

def add_dispersion(x,magnitude):
    r=np.random.uniform(0,magnitude,*x.shape)
    for i in range(len(r)):
        r[i]*=1 if (np.random.randint(0,2)) else -1

    return( x + r)

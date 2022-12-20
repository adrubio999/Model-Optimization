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

# Por defecto ascending=True: la escala va de mas oscuro hasta el color: más claro mayor F1
# Si ascending=False, la escala es de más claro a mas oscuro: mas oscuro mayor F1
def define_colors(var,n,arq,ascending=True):
    colors=[]
    colors2=[]
    alpha=[]
    alpha2=[]
    cmap_grey=mpl.colormaps['Greys']  
    var=np.array(var,dtype=float)
    # Custom colors depending on var
    inds=(-var).argsort()
    var=var[inds]
    max_ind=np.max(var[:n])
    min_ind=np.min(var[:n])
    
    for i,F1 in enumerate(var):
        if i <n:
            #if arq==''
            if ascending==True:
                scaling=(F1-min_ind)/(max_ind-min_ind)*0.5+0.5
            else:
                scaling=(max_ind-F1)/(max_ind-min_ind)*0.5+0.5
            #colors.append(cmap_blue((F1-min_ind)/(max_ind-min_ind)*0.5+0.5))
            a=tuple([x*scaling for x in list(colors_dic[arq])])
            colors.append(a)
            alpha.append(1)
        else:
            colors2.append(cmap_grey(0.4))
            alpha2.append(0.1)
    # The colors are returned in decreasing order with respect to var

    return colors+colors2,alpha+alpha2,inds

def add_dispersion(x,magnitude):
    r=np.random.uniform(0,magnitude,*x.shape)
    for i in range(len(r)):
        r[i]*=1 if (np.random.randint(0,2)) else -1

    return( x + r)

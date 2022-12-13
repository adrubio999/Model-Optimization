import numpy as np
import matplotlib as mpl
def define_colors(var,n):
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
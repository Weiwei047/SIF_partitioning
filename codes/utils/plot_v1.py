import warnings
warnings.simplefilter(action='ignore')
import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import tensorflow.keras.backend as K

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 


def regression_line(x,y,ax,xlim,RMSE_4dig=False,num_seq=None,
                    color1='black',color2='saddlebrown',fontsize=12,
                    xy=(.05,.65),legend=False,diag=True,R2_3digt=False):
    # remove NAN values in x, y series
    x,y = x.to_frame(), y.to_frame()
    x_y = pd.concat([x,y],axis=1)
    
    if x_y.isnull().values.any():
        x_y = x_y.dropna()
    
    x, y = x_y.iloc[:,0],x_y.iloc[:,1]
    
    # plot diagonal line
    min_value,max_value = xlim
    diag_line = np.linspace(min_value, max_value, 100)
    if diag:
        
        ax.plot(diag_line, diag_line,'--',color=color1,label='1:1 Line')
    
    # regression & regression line
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    RMSE = (((y - x) ** 2).mean()) ** .5
    RMSPE = ((((y - x)/y) ** 2).mean()) ** .5 * 100 #(np.sqrt(np.mean(np.square((y - x) / y)))) * 100
    R2 = r_value**2
    
    
    y_pre = diag_line*slope+intercept
    ax.plot(diag_line, y_pre,color=color2,label='Regression Line')
 
    if num_seq is not None:
        ax.annotate(num_seq,xy=(.05,.9), xycoords='axes fraction',fontweight='bold',fontsize=13)
    if legend:
        ax.legend(loc=4,frameon=False)
        
    dig_var = '%.4f' if RMSE_4dig else '%.2f'
    dig_R2 = '%.3f' if R2_3digt else '%.2f'
        
    if intercept >=0:
        ax.annotate(
                    f'$R^{2} = $'+dig_R2 % R2 + 
                    f'\n$RMSE = $'+dig_var % RMSE      +
                    f'\n$y = $%.2f$x + $%.2f'%(slope,intercept), 
                    xy=xy, xycoords='axes fraction',fontsize=fontsize)
    else:
        ax.annotate(
                    f'$R^{2} = $'+dig_R2 % R2 + 
                    f'\n$RMSE = $'+dig_var % RMSE      +
                    f'\n$y = $%.2f$x - $%.2f'%(slope,np.abs(intercept)), 
                    xy=xy, xycoords='axes fraction',fontsize=fontsize)

            
            
def GPP_SIF_dependence(data,SIF_var,GPP_var,xlim,ylim,ax,
                       c,cmap,fig,cf_label,s,
                       vmin=None,vmax=None,resort=False,ascending=True,shrink=0.8):
    
    if resort:
        data = data.sort_values(by=c,ascending=ascending)
    
    cf = ax.scatter(data[SIF_var],data[GPP_var],c=data[c],cmap=cmap,vmin=vmin,vmax=vmax,s=s)
    fig.colorbar(cf, ax=ax,extend='max',label=cf_label,shrink=shrink)
    regression_line_GPPSIF(data[SIF_var],data[GPP_var],ax=ax,xy=(.7,.15))
 

    
def regression_line_GPPSIF(x, y, ax, num_seq=None,SIF=False, 
                           xy = (0.6,0.76), color='black', opt='-', col='black'):
    
    # remove NAN values in x, y series
    x,y = x.to_frame(), y.to_frame()
    x_y = pd.concat([x,y],axis=1)
    
    if x_y.isnull().values.any():
        x_y = x_y.dropna()
    
    x, y = x_y.iloc[:,0],x_y.iloc[:,1]
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    max_value,min_value = max(x),min(x)
        
    line  = np.linspace(min_value, max_value, 100)
    y_pre = line*slope+intercept
    
    line = line[y_pre<max(y)]
    y_pre = y_pre[y_pre<max(y)]
    
    ax.plot(line, y_pre,opt,color=color,linewidth=1.5)
    if num_seq is not None:
        ax.annotate(num_seq,xy=(.05,.9), xycoords='axes fraction',fontweight='bold',fontsize=13)
        
    if intercept > 0:
        ax.annotate('$R^{2}$='+' {:.2f}'.format(r_value**2), 
                    xy=xy, color=col, xycoords='axes fraction')
    elif intercept < 0:
        ax.annotate('$R^{2}$='+' {:.2f}'.format(r_value**2), 
                    xy=xy, color=col, xycoords='axes fraction')
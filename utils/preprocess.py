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


def include_predictions(data,NEE_NN,GPP_NN,Reco_NN,SIF_NN):
    
    data['SIF_NN'] = SIF_NN
    data['NEE_NN'] = NEE_NN
    data['GPP_NN'] = GPP_NN
    data['Reco_NN'] = Reco_NN
    
    return data

def div_daynight(data):
    
    day = data[data.APAR_label==1]
    night = data[data.APAR_label==0]
    
    return day, night


def standard_x(x_train,x_test=None):
    # the mean and std values are only calculated by training set
    x_mean   = x_train.mean(axis=0) 
    x_std    = x_train.std(axis=0)
    x_train1 = ((x_train-x_mean)/x_std).values
    if x_test is not None:
        x_test1  = ((x_test -x_mean)/x_std).values
        return x_train1, x_test1
    return x_train1 


def split_train_test(dataset,seed,EV1_label,EV2_label,EV3_label,var_NEE,var_SIF,
                     flux_label=['GPP_canopy','Reco_canopy','Reco_NT','GPP_NT','Reco_DT','GPP_DT']):  
    
    
    train, test    = train_test_split(dataset, test_size=0.3, random_state=seed,shuffle=True)
    train['train_label'] = 'Training set'
    test['train_label']  = 'Test set'


    EV1_train = train[EV1_label].astype('float32')# EV for GPP
    EV2_train = train[EV2_label].astype('float32')# EV for ER
    EV3_train = train[EV3_label].astype('float32')# EV for SIF
    NEE_train = train[var_NEE].astype('float32')
    SIF_train = train[var_SIF].astype('float32')
    flux_train= train[flux_label].astype('float32')
    label_train= train['APAR_label'].astype('float32').values


    EV1_test  = test[EV1_label].astype('float32')# EV for GPP
    EV2_test  = test[EV2_label].astype('float32')# EV for Reco
    EV3_test  = test[EV3_label].astype('float32')# EV for SIF
    NEE_test  = test[var_NEE].astype('float32')
    SIF_test  = test[var_SIF].astype('float32')
    flux_test = test[flux_label].astype('float32')
    label_test = test['APAR_label'].astype('float32').values
    
    # standardization
    EV1_train1,EV1_test1 = standard_x(EV1_train,EV1_test)
    EV2_train1,EV2_test1 = standard_x(EV2_train,EV2_test)
    EV3_train1,EV3_test1 = standard_x(EV3_train,EV3_test)
    
    NEE_norm = (np.abs(NEE_train.values)).max()
    NEE_train1 = NEE_train.values/NEE_norm
    SIF_norm = (np.abs(SIF_train.values)).max()
    SIF_train1 = SIF_train.values/SIF_norm

    train_dataset = (train,EV1_train1,EV2_train1,EV3_train1,NEE_train1,SIF_train1,flux_train,label_train)
    test_dataset  = (test, EV1_test1, EV2_test1, EV3_test1, NEE_test, SIF_test, flux_test, label_test)
    
    return train_dataset, test_dataset, NEE_norm, SIF_norm


def regression_line(x,y,ax,xlim,RMSE_4dig=False,num_seq=None,
                    color1='black',color2='saddlebrown',fontsize=12,
                    xy=(.05,.65),legend=False,diag=True,only_R2=False,R_appro=None,R2_3digt=False):
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
    
    y_pre = diag_line*slope+intercept
    ax.plot(diag_line, y_pre,color=color2,label='Regression Line')
    
    if R_appro is not None:
        R2 = R_appro
    else:
        R2 = r_value**2
    if num_seq is not None:
        ax.annotate(num_seq,xy=(.05,.9), xycoords='axes fraction',fontweight='bold',fontsize=13)
    if legend:
        ax.legend(loc=4,frameon=False)
    if RMSE_4dig:
        dig_var = '%.4f'
    else:
        dig_var = '%.2f'
        
    if R2_3digt:
        dig_R2 = '%.3f'
    else:
        dig_R2 = '%.2f'
        
    if only_R2:
        ax.annotate(
                    f'$R^{2} = $'+"%.2f" % R2,
                    xy=xy, xycoords='axes fraction',fontsize=fontsize)
    else:
        
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
            
            
            
def GPP_SIF_dependence(data,SIF_var,GPP_var,xlim,ylim,ax,c,cmap,fig,cf_label,s,vmin=None,vmax=None,resort=False,ascending=True,shrink=0.8):
    
    if resort:
        data = data.sort_values(by=c,ascending=ascending)
    
    cf = ax.scatter(data[SIF_var],data[GPP_var],c=data[c],cmap=cmap,vmin=vmin,vmax=vmax,s=s)
    fig.colorbar(cf, ax=ax,extend='max',label=cf_label,shrink=shrink)
    regression_line_GPPSIF(data[SIF_var],data[GPP_var],ax=ax,xy=(.7,.15))
 
    
def regression_line_GPPSIF(x, y, ax, num_seq=None,SIF=False, xy = (0.6,0.76), color='black', opt='-', col='black'):
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
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




def standard_x(x_train,x_test=None):
    
    '''
    This is for input standardization: X' = X - mean(X)/std(X)
    '''
    
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
    
    '''
    This is to split the original dataset into training & testing sets
    
    
    Parameters
    ----------  
    dataset:   the original dataset
    seed:      the random seed that ensures the same results everytime we split the data 
    EV1_label: variables used for the prediction of gross primary productivity (GPP)
    EV2_label: variables used for the prediction of ecosystem respiration (Reco)
    EV3_label: variables used for the prediction of the GPP-SIF relationship 
    var_NEE:   column name for the observed NEE
    var_SIF:   column name for the observed SIF
    flux_label: columns corresponds to SCOPE-simulated GPP & ER ("truth" value here), NT- and DT- estimated GPP & ER
    
    
    Returns
    ----------  
    train_dataset: the splitted training set 
    test_dataset:  the splitted test set
    NEE_norm:      the normalized NEE
    SIF_norm:      the normalized SIF
    '''
    
    
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



def include_predictions(data,NEE_NN,GPP_NN,Reco_NN,SIF_NN):
    
    data['SIF_NN'] = SIF_NN
    data['NEE_NN'] = NEE_NN
    data['GPP_NN'] = GPP_NN
    data['Reco_NN'] = Reco_NN
    
    return data


def div_daynight(data):
    
    day = data[data.PAR_label==1]
    night = data[data.PAR_label==0]
    
    return day, night



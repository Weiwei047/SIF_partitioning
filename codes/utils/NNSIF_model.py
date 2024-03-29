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
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Lambda, Activation, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.constraints import NonNeg
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import Constant
    
    
    
def get_prediction_model(n_neuron,input1_shape,input2_shape,input3_shape,SIF_act=None,SIF_log=False):
    
    '''
    This is to define the structure of the NN_SIF model
    
    Parameters
    ---------- 
    n_neuron:     the number of neurons for each hidden layer
    input1_shape: the shape of the input layer for GPP
    input2_shape: the shape of the input layer for Reco
    input3_shape: the shape of the input layer for SIF
    SIF_act:      the activation function for the SIF prediction
    SIF_log:      whether to use a logarithm function for the SIF prediction
    
    Returns:
    ----------
    model: the defined NN_SIF model
    
    '''
    
    # GPP
    APAR_input= Input(shape=(1,), dtype='float32', name='APAR_input')
    EV_input1 = Input(shape=(input1_shape,), dtype='float32', name='EV_input1')
    x         = Dense(n_neuron, activation='relu',name='hidden1_1')(EV_input1)
    x         = Dense(n_neuron, activation='relu',name='hidden1_2')(x)
    ln_GPP    = Dense(1, activation = None, name='ln_GPP')(x)
    GPP_1     = Lambda(lambda x: K.exp(x),  name='GPP_1')(ln_GPP)
    GPP       = keras.layers.Multiply(name='GPP')([GPP_1,APAR_input])

    # Reco
    EV_input2 = Input(shape=(input2_shape,), dtype='float32', name='EV_input2')
    x         = Dense(n_neuron, activation='relu',name='hidden2_1')(EV_input2) 
    x         = Dense(n_neuron, activation='relu',name='hidden2_2')(x) 
    ln_Reco   = Dense(1,  activation=None,name='ln_Reco')(x)
    Reco      = Lambda(lambda x: K.exp(x),name='Reco')(ln_Reco)

    NEE       = keras.layers.Subtract(name='NEE')([Reco, GPP])
    
    # SIF = f(GPP,other environmental variables)
    EV_input3 = Input(shape=(input3_shape,), dtype='float32', name='EV_input3')
    combined  = keras.layers.Concatenate(name='combined')([GPP, EV_input3])
    x         = Dense(n_neuron, activation='relu',name='hidden3_1')(combined)
    x         = Dense(n_neuron, activation='relu',name='hidden3_2')(x)
    if SIF_log:
        ln_SIF    = Dense(1,  activation=None,name='ln_SIF')(x)
        SIF       = Lambda(lambda x: K.exp(x),name='SIF')(ln_SIF)
    else:
        SIF       = Dense(1, name='SIF',activation=SIF_act)(x)
    
    model = Model(inputs=[APAR_input,EV_input1,EV_input2,EV_input3], outputs=[NEE,SIF])


    return model


# Custom loss layer
# reference: https://github.com/yaringal/multi-task-learning-example/blob/master/multi-task-learning-example.ipynb
class CustomMultiLossLayer(Layer):
    
    '''
    This is to achieve the multi-task learning
    '''
    
    def __init__(self, nb_logvars=2,nb_outputs=2, **kwargs):
        self.nb_outputs = nb_outputs
        self.nb_logvars = nb_logvars
        self.is_placeholder = True
        self.mse_nee = tf.keras.metrics.Mean(name='mse_nee')
        self.mse_sif = tf.keras.metrics.Mean(name='mse_sif')
        super(CustomMultiLossLayer, self).__init__(**kwargs)
        
    def build(self, input_shape=None):
        # initialise log_vars
        self.log_vars = []
        for i in range(self.nb_logvars):
            self.log_vars += [self.add_weight(name='log_var' + str(i), shape=(1,),
                                              initializer=Constant(0.), trainable=True)]
        super(CustomMultiLossLayer, self).build(input_shape)


    def multi_loss(self, NEE_true,SIF_true,NEE_pred,SIF_pred):
        loss = 0
        if self.nb_logvars == 2:
            log_var1, log_var2 = self.log_vars
            precision_NEE = K.exp(-log_var1)
            precision_SIF = K.exp(-log_var2)
            
            loss = K.sum(precision_NEE * (NEE_true - NEE_pred)**2. + log_var1, -1) + \
                   K.sum(precision_SIF * (SIF_true - SIF_pred)**2. + log_var2, -1) 
              
        return K.mean(loss) 

    def call(self, inputs):
        # metrics: MSE of NEE & SIF
        NEE_true, SIF_true  = inputs[:self.nb_outputs] 
        NEE_pred, SIF_pred  = inputs[self.nb_outputs:2*self.nb_outputs]
        APAR_input = inputs[-1]
        
        # MSE_NEE = K.mean(K.square(NEE_pred-NEE_true))
        MSE_NEE = self.mse_nee(tf.square(NEE_pred-NEE_true)) # tf.square(NEE_pred-NEE_true)
        MSE_SIF = self.mse_sif(K.mean(K.square(SIF_pred*APAR_input-SIF_true)))
        
        self.add_metric(MSE_NEE)#,  name="MSE_NEE"
        self.add_metric(MSE_SIF) #,  name="MSE_SIF"
        
        
        if self.nb_logvars ==2:
            loss    = self.multi_loss(NEE_true,SIF_true,NEE_pred,SIF_pred)
        
        self.add_loss(loss, inputs=inputs)
        
        return K.concatenate(inputs, -1)


def get_trainable_model(prediction_model,input1_shape,input2_shape,input3_shape):
    
    '''
    This is to implement the multi-task training for NN_SIF: the weights of NEE & SIF are inferred purely from data
    
    Parameters
    ---------- 
    prediction_model: the defined structure of NN_SIF (the returned variable of the 'get_prediction_model' function)
    input1_shape: the shape of the input layer for GPP
    input2_shape: the shape of the input layer for Reco
    input3_shape: the shape of the input layer for SIF
    
    Returns:
    ----------
    model: the final NN_SIF model
    
    '''
    
    APAR_input= Input(shape=(1,), dtype='float32', name='APAR_input')
    EV_input1 = Input(shape=(input1_shape,), dtype='float32', name='EV_input1')
    EV_input2 = Input(shape=(input2_shape,), dtype='float32', name='EV_input2')
    EV_input3 = Input(shape=(input3_shape,), dtype='float32', name='EV_input3')
    
    NEE_pred,SIF_pred = prediction_model([APAR_input,EV_input1,EV_input2,EV_input3])
    
    NEE_true = Input(shape=(1,), name='NEE_true')
    SIF_true = Input(shape=(1,), name='SIF_true')
    
    out      = CustomMultiLossLayer(nb_outputs=2)([NEE_true,SIF_true,NEE_pred,SIF_pred,APAR_input])
    return Model([APAR_input,EV_input1,EV_input2,EV_input3,NEE_true,SIF_true], out)


def layer_output(model,layer_name, label_test,test_input1,test_input2,test_input3):
    
    '''
    To retrieve the output predicted by a specified layer
    
    Parameters
    ---------- 
    model:      the well-trained model
    layer_name: the layer of which output we want to obtain  
    label_test: the 'label' input 
    test_input1: the input data for GPP estimation
    test_input2: the input data for Reco estimation
    test_input3: the input data for the estimation of the GPP-SIF relationship
    
    Returns:
    ---------- 
    inter_output: the predicted output of the specifed layer
    
    '''
    
    layer_model  = Model(inputs =model.input,
                         outputs=model.get_layer(layer_name).output)
    inter_output = layer_model.predict({'APAR_input': label_test,
                                        'EV_input1': test_input1,
                                        'EV_input2': test_input2,
                                        'EV_input3': test_input3})
    return inter_output


def fluxes_SIF_predict(model,label,EV1, EV2, EV3, NEE_max_abs, SIF_max_abs):
    
    '''
    To predict GPP, Reco & SIF using the well-trained NN_SIF model
    
    Parameters
    ----------
    model: the well-trained NN_SIF model
    label: the 'label' input
    EV1:   the input data for the GPP estimation
    EV2:   the input data for the Reco estimation
    EV3:   the input data for the estimation of the GPP-SIF relationship
    NEE_max_abs: the max value of NEE, which is used to convert back to the real value of NEE
    SIF_max_abs: the max value of SIF, which is used to convert back to the real value of SIF
    
    Returns:
    ---------- 
    NEE_NN:  the NN-predicted NEE 
    GPP_NN:  the NN-predicted GPP 
    Reco_NN: the NN-predicted Reco 
    SIF_NN:  the NN-predicted SIF
    '''
    
    NEE_NN  = (layer_output(model,'NEE',  label,EV1,EV2,EV3) * NEE_max_abs)
    NEE_NN  = NEE_NN.reshape(NEE_NN.shape[0],)
    
    GPP_NN  = (layer_output(model,'GPP',  label,EV1,EV2,EV3) * NEE_max_abs).reshape(NEE_NN.shape[0],)
    Reco_NN = (layer_output(model,'Reco', label,EV1,EV2,EV3) * NEE_max_abs).reshape(NEE_NN.shape[0],)
    SIF_NN  = (layer_output(model,'SIF',  label,EV1,EV2,EV3) * SIF_max_abs).reshape(NEE_NN.shape[0],)
    

    return NEE_NN, GPP_NN, Reco_NN, SIF_NN
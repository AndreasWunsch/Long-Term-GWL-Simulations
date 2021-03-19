
#reproducability
from numpy.random import seed
seed(1+347823)
import tensorflow as tf
tf.random.set_seed(1+63493)

import numpy as np
import os
import pandas as pd
import datetime
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import shap

gpus = tf.config.experimental.list_physical_devices('GPU')


def load_GW_and_HYRAS_Data(Well_ID):
    #define where to find the data
    pathGW = "./GWData"
    pathHYRAS = "./HYRAS"
    pathconnect = "/"
    
    #load and merge the data
    GWData = pd.read_csv(pathGW+pathconnect+Well_ID+'_GW-Data.csv', 
                         parse_dates=['Date'],index_col=0, dayfirst = True, 
                         decimal = '.', sep=',')
    HYRASData = pd.read_csv(pathHYRAS+pathconnect+Well_ID+'_weeklyData_HYRAS.csv',
                            parse_dates=['Date'],index_col=0, dayfirst = True,
                            decimal = '.', sep=',')
    data = pd.merge(GWData, HYRASData, how='inner', left_index = True, right_index = True)

    return data

def split_data(data, GLOBAL_SETTINGS):
    dataset = data[(data.index < GLOBAL_SETTINGS["test_start"])] #Testdaten abtrennen
    
    TrainingData = dataset
    
    TestData = data[(data.index >= GLOBAL_SETTINGS["test_start"]) & (data.index <= GLOBAL_SETTINGS["test_end"])] #Testdaten entsprechend dem angegebenen Testzeitraum
    TestData_ext = pd.concat([dataset.iloc[-GLOBAL_SETTINGS["seq_length"]:], TestData], axis=0) # extend Testdata to be able to fill sequence later                                              

    return TrainingData, TestData, TestData_ext

def to_supervised(data, GLOBAL_SETTINGS):
    X, Y = list(), list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + GLOBAL_SETTINGS["seq_length"]
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = data[i:end_idx, 1:], data[end_idx, 0]
        X.append(seq_x)
        Y.append(seq_y)
    return np.array(X), np.array(Y)

# =============================================================================
#### start
# =============================================================================

with tf.device("/gpu:0"):
    
    time1 = datetime.datetime.now()
    basedir = './'
    os.chdir(basedir)
    
    well_list = pd.read_csv("./list.txt")

# =============================================================================
#### loop 
# =============================================================================

    for pp in range(well_list.shape[0]):

        Well_ID = well_list.ID[pp]
        print(str(pp)+": "+Well_ID)
        
        bestkonfig = pd.read_csv('./log_summary_CNN_'+Well_ID+'.txt',delimiter='=',skiprows=(10),nrows=(7),header = None)
        bestkonfig.columns = ['hp','value']
        filters_int = int(bestkonfig.value[0])
        densesize_int = int(bestkonfig.value[1])
        seqlength_int = int(bestkonfig.value[2])
        batchsize_int = int(bestkonfig.value[3])

        
        pathGW = "./GWData/"
        GWData = pd.read_csv(pathGW+Well_ID+'_GW-Data.csv',parse_dates=['Date'],index_col=0, dayfirst = True,decimal = '.', sep=',')
        GWData = GWData[(GWData.index <= pd.to_datetime('01012016', format='%d%m%Y'))]


        GLOBAL_SETTINGS = {
            'pp': Well_ID,
            'batch_size': batchsize_int, #16-128
            'kernel_size': 3, #ungerade
            'dense_size': densesize_int, 
            'filters': filters_int, 
            'seq_length': seqlength_int,
            'clip_norm': True,
            'clip_value': 1,
            'epochs': 100,
            'learning_rate': 1e-3,
            'test_start': pd.to_datetime('02012012', format='%d%m%Y'),
            'test_end': pd.to_datetime('28122015', format='%d%m%Y')
        }
        
        ## load data
        data = load_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])
        
        if GLOBAL_SETTINGS["test_end"] > data.index[-1]:
            GLOBAL_SETTINGS["test_end"] = data.index[-1]
            GLOBAL_SETTINGS["test_start"] = GLOBAL_SETTINGS["test_end"] - datetime.timedelta(days=(365*4))
        
            
        #scale data
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
        scaler_gwl.fit(pd.DataFrame(data['GWL']))
        data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)
        
        #split data
        TrainingData_n, TestData_n, TestData_ext_n = split_data(data_n, GLOBAL_SETTINGS)
        
        #sequence data
        X_train, Y_train = to_supervised(TrainingData_n.values, GLOBAL_SETTINGS)
        X_test, Y_test = to_supervised(TestData_ext_n.values, GLOBAL_SETTINGS) 
        
        path = './nets_noMCDropout/'+Well_ID;
        
        inimax = 10

        for ini in range(inimax):
            print("ini: "+str(ini))
            model = tf.keras.models.load_model(path+"/ini"+str(ini))
            
# =============================================================================
#### SHAP
# =============================================================================

            background = X_train
            e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(X_test)
            
            
            np.savetxt('./SHAP_values_CNN_'+Well_ID+'_ini'+str(ini)+'_P.txt',
                        shap_values[0][:,:,0], fmt='%.6f', delimiter=';', newline='\n')
            np.savetxt('./SHAP_values_CNN_'+Well_ID+'_ini'+str(ini)+'_T.txt',
                        shap_values[0][:,:,1], fmt='%.6f', delimiter=';', newline='\n')
            

            shap_vals = np.asarray(shap_values[0])
            shap_vals = shap_vals.reshape(-1, shap_vals.shape[-1])
            
            if ini == 0:
                s = shap_vals
                X = X_test.reshape(-1, X_test.shape[-1])
            else:
                s = np.append(s,shap_vals,0)
                X = np.append(X,X_test.reshape(-1, X_test.shape[-1]),0)
            
        np.savetxt('./Xvalues_for_SHAP_CNN_'+Well_ID+'_P.txt',
                   X_test[:,:,0], fmt='%.6f', delimiter=';', newline='\n')
        np.savetxt('./Xvalues_for_SHAP_CNN_'+Well_ID+'_T.txt',
                   X_test[:,:,1], fmt='%.6f', delimiter=';', newline='\n')
            
        shap.summary_plot(s, X,feature_names=['P', 'T'], show=False)
        pyplot.title(Well_ID)
        pyplot.xlabel("SHAP value (impact on GWL)")
        pyplot.savefig('./'+Well_ID+'_SHAP_Summary.png', dpi=600,bbox_inches='tight') 
        pyplot.show()
        
        
        # shap.dependence_plot(0,s, X,feature_names=['P', 'T'], interaction_index=None)
        # shap.dependence_plot(0,s, X,feature_names=['P', 'T'])
        # shap.dependence_plot(1,s, X,feature_names=['P', 'T'], interaction_index=None)
        # shap.dependence_plot(1,s, X,feature_names=['P', 'T'])
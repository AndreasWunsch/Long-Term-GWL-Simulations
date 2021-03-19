
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
from uncertainties import unumpy

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
    dataset = data[(data.index < GLOBAL_SETTINGS["'simulation_start'"])] 
    
    TrainingData = dataset[0:round(0.8 * len(dataset))]
    StopData = dataset[round(0.8 * len(dataset))+1:]
    StopData_ext = dataset[round(0.8 * len(dataset))+1-GLOBAL_SETTINGS["seq_length"]:] #extend data according to delays/sequence length

    return TrainingData, StopData, StopData_ext

def to_supervised(data, GLOBAL_SETTINGS):
    #make the data sequential
    #modified after Jason Brownlee and machinelearningmastery.com
    
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

def to_supervised_onlyX(data, GLOBAL_SETTINGS):
    #make the data sequential
    #modified after Jason Brownlee and machinelearningmastery.com
    
    X = list()
    # step over the entire history one time step at a time
    for i in range(len(data)):
        # find the end of this pattern
        end_idx = i + GLOBAL_SETTINGS["seq_length"]
        # check if we are beyond the dataset
        if end_idx >= len(data):
            break
        # gather input and output parts of the pattern
        seq_x = data[i:end_idx, :]
        X.append(seq_x)
    return np.array(X)

def generate_scalers(pp):
    # load training data again, to use same scalers
    data_orig = load_GW_and_HYRAS_Data(pp)
        
    #scale data
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data_orig['GWL']))
    data_orig.drop(columns='GWL', inplace=True)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(data_orig)
    
    return scaler, scaler_gwl

class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
    
def gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train,X_stop, Y_stop):
    # define models
    seed(ini+872527)
    tf.random.set_seed(ini+87747)
    
    inp = tf.keras.Input(shape=(GLOBAL_SETTINGS["seq_length"], X_train.shape[2]))
    cnn = tf.keras.layers.Conv1D(filters=GLOBAL_SETTINGS["filters"],
                                         kernel_size=GLOBAL_SETTINGS["kernel_size"],
                                         activation='relu',
                                         padding='same')(inp)
    cnn = tf.keras.layers.MaxPool1D(padding='same')(cnn)
    cnn = MCDropout(0.5)(cnn)
    cnn = tf.keras.layers.Flatten()(cnn)
    cnn = tf.keras.layers.Dense(GLOBAL_SETTINGS["dense_size"], activation='relu')(cnn)
    output1 = tf.keras.layers.Dense(1, activation='linear')(cnn)
    
    # tie together
    model = tf.keras.Model(inputs=inp, outputs=output1)
        
    optimizer = tf.keras.optimizers.Adam(learning_rate=GLOBAL_SETTINGS["learning_rate"], epsilon=10E-3, clipnorm=GLOBAL_SETTINGS["clip_norm"])
    
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    
    # early stopping
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=15,restore_best_weights = True)
    
    # fit network
    history = model.fit(X_train, Y_train, validation_data=(X_stop, Y_stop), epochs=GLOBAL_SETTINGS["epochs"], verbose=0,
                        batch_size=GLOBAL_SETTINGS["batch_size"], callbacks=[es])
    
    return model, history

def predict_distribution(X, model, n):
    preds = [model(X) for _ in range(n)]
    return np.hstack(preds)

def train_and_save_model(Well_ID,densesize_int, seqlength_int, batchsize_int, filters_int):
    
    GLOBAL_SETTINGS = {
        'pp': Well_ID,
        'batch_size': batchsize_int,
        'kernel_size': 3, 
        'dense_size': densesize_int, 
        'filters': filters_int, 
        'seq_length': seqlength_int,
        'clip_norm': True,
        'clip_value': 1,
        'epochs': 100,
        'learning_rate': 1e-3,
        'simulation_start': pd.to_datetime('16092013', format='%d%m%Y')
    }

    ## load data
    data = load_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])
        
    #scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl = MinMaxScaler(feature_range=(-1, 1))
    scaler_gwl.fit(pd.DataFrame(data['GWL']))
    data_n = pd.DataFrame(scaler.fit_transform(data), index=data.index, columns=data.columns)

    #split data
    TrainingData, StopData, StopData_ext = split_data(data, GLOBAL_SETTINGS)
    TrainingData_n, StopData_n, StopData_ext_n = split_data(data_n, GLOBAL_SETTINGS)
    
    #sequence data
    X_train, Y_train = to_supervised(TrainingData_n.values, GLOBAL_SETTINGS)
    X_stop, Y_stop = to_supervised(StopData_ext_n.values, GLOBAL_SETTINGS) 

    #build and train model with different initializations
    inimax = 10
    path = './'+Well_ID; #path for saving the models
    
    if os.path.isdir(path) == False:
            os.mkdir(path)
            
    f = open(path+'traininghistory_CNN_'+Well_ID+'.txt', "w")
    
    for ini in range(inimax):
        if os.path.isdir(path+"/ini"+str(ini)) == False:
            
            print(str(pp)+": "+Well_ID+"_ini"+str(ini))
            model, history = gwmodel(ini,GLOBAL_SETTINGS,X_train, Y_train, X_stop, Y_stop)  
            
            model.save(path+"/ini"+str(ini))
            
            loss = np.zeros((1, 100))
            loss[:,:] = np.nan
            loss[0,0:np.shape(history.history['loss'])[0]] = history.history['loss']
            val_loss = np.zeros((1, 100))
            val_loss[:,:] = np.nan
            val_loss[0,0:np.shape(history.history['val_loss'])[0]] = history.history['val_loss']
            print('loss', file = f)
            print(loss.tolist(), file = f)
            print('val_loss', file = f)
            print(val_loss.tolist(), file = f)

        else: print(Well_ID+"_ini"+str(ini)+" - already exists")
    
    f.close()
    return GLOBAL_SETTINGS

def load_proj_data(proj_name,Well_ID,GLOBAL_SETTINGS):
    
    data1 = load_GW_and_HYRAS_Data(GLOBAL_SETTINGS["pp"])
    data1 = data1.drop(columns='GWL')

    path = "./"+Well_ID+"_weeklyData_"+proj_name+".csv"
    proj_data = pd.read_csv(path,parse_dates=['Date'],index_col=0, dayfirst = True, 
                         decimal = '.', sep=',')
    proj_data = proj_data[(proj_data.index >= GLOBAL_SETTINGS["'simulation_start'"])]
    proj_data_ext = pd.concat([data1.iloc[-GLOBAL_SETTINGS["seq_length"]:], proj_data], axis=0)
    
    return proj_data, proj_data_ext

def applymodels_to_climproj(Well_ID,densesize_int, seqlength_int, batchsize_int, filters_int, proj_name):
    
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
        'simulation_start': pd.to_datetime('16092013', format='%d%m%Y')
    }

    data, data_ext = load_proj_data(proj_name,Well_ID,GLOBAL_SETTINGS)
        
    #scale data
    scaler,scaler_gwl = generate_scalers(GLOBAL_SETTINGS["pp"])
    data_ext_n = pd.DataFrame(scaler.transform(data_ext), index=data_ext.index, columns=data_ext.columns)

    #sequence data
    X_test = to_supervised_onlyX(data_ext_n.values, GLOBAL_SETTINGS)
    
    #model path
    path = './'+Well_ID;
    
    inimax = 10
    sim_members = np.zeros((len(X_test), inimax))
    sim_members[:] = np.nan
    
    sim_std = np.zeros((len(X_test), inimax))
    sim_std[:] = np.nan
    
    for ini in range(inimax):
        loaded_model = tf.keras.models.load_model(path+"/ini"+str(ini))
        y_pred_distribution = predict_distribution(X_test, loaded_model, 100)
        sim = scaler_gwl.inverse_transform(y_pred_distribution)
        sim_members[:, ini], sim_std[:, ini]= sim.mean(axis=1), sim.std(axis=1)

    sim_members_uncertainty = unumpy.uarray(sim_members,1.96*sim_std) #1.96 because of sigma rule for 95% confidence
    
    sim_mean = np.nanmedian(sim_members,axis = 1)
    
    sim_mean_uncertainty = np.sum(sim_members_uncertainty,axis = 1)/inimax
        
    return sim_mean, sim_members, sim_mean_uncertainty, sim_members_uncertainty, data, inimax

# =============================================================================
#### start
# =============================================================================

with tf.device("/gpu:0"):
    
    time1 = datetime.datetime.now()
    basedir = './'
    os.chdir(basedir)
    
    well_list = pd.read_csv("./list.txt")

# =============================================================================
#### training loop
# =============================================================================
    for pp in range(well_list.shape[0]):
        time_single = datetime.datetime.now()
        seed(1)
        tf.random.set_seed(1)
    
        Well_ID = well_list.ID[pp]
        
        # read optimized hyperparameters
        bestkonfig = pd.read_csv('./log_summary_CNN_'+Well_ID+'.txt',delimiter='=',skiprows=(10),nrows=(7),header = None)
        bestkonfig.columns = ['hp','value']
        filters_int = int(bestkonfig.value[0])
        densesize_int = int(bestkonfig.value[1])
        seqlength_int = int(bestkonfig.value[2])
        batchsize_int = int(bestkonfig.value[3])
        
        
        train_and_save_model(Well_ID,densesize_int, seqlength_int, batchsize_int, filters_int)
        
# =============================================================================
#### projection loop    
# =============================================================================
    
    Projections = ["CCCma-CanESM2_rcp85_r1i1p1_CLMcom-CCLM4-8-17","ICHEC-EC-EARTH_rcp85_r1i1p1_KNMI-RACMO22E","MIROC-MIROC5_rcp85_r1i1p1_GERICS-REMO2015","MOHC-HadGEM2-ES_rcp85_r1i1p1_CLMcom-CCLM4-8-17","MPI-M-MPI-ESM-LR_rcp85_r1i1p1_UHOH-WRF361H","MPI-M-MPI-ESM-LR_rcp85_r2i1p1_MPI-CSC-REMO2009"]
    
    for pp in range(well_list.shape[0]):
        Well_ID = well_list.ID[pp]
        
        if not os.path.exists(Well_ID):
            os.makedirs(Well_ID)
        
        # read optimized hyperparameters
        bestkonfig = pd.read_csv('./log_summary_CNN_'+Well_ID+'.txt',delimiter='=',skiprows=(10),nrows=(7),header = None)
        bestkonfig.columns = ['hp','value']
        filters_int = int(bestkonfig.value[0])
        densesize_int = int(bestkonfig.value[1])
        seqlength_int = int(bestkonfig.value[2])
        batchsize_int = int(bestkonfig.value[3])
        
        #read observed GW Data again for plotting
        pathGW = "./GWData/"
        GWData = pd.read_csv(pathGW+Well_ID+'_GW-Data.csv',parse_dates=['Date'],index_col=0, dayfirst = True,decimal = '.', sep=',')
        
        for proj in range(len(Projections)):
            proj_name = Projections[proj]
            print(str(pp)+": "+Well_ID+" - "+proj_name)
            sim_mean, sim_members, sim_mean_uncertainty, sim_members_uncertainty, data, inimax = applymodels_to_climproj(Well_ID,densesize_int, seqlength_int, batchsize_int, filters_int, rH, T, Tsin, proj_name)
            
# =============================================================================
#### plot Test-Section
# =============================================================================

            y_err = unumpy.std_devs(sim_mean_uncertainty)
            
            pyplot.figure(figsize=(20,6))
            pyplot.fill_between(data.index, sim_mean.reshape(-1,) - y_err,
                                sim_mean.reshape(-1,) + y_err, facecolor = (1,0.7,0,0.5),
                                label ='95% confidence',linewidth = 1,
                                edgecolor = (1,0.7,0,0.6))            
            pyplot.plot(data.index, sim_mean, 'r', label ="simulated median", linewidth = 1)
            pyplot.plot(GWData.index, GWData['GWL'], 'k', label ="observed", linewidth=1,alpha=0.9)
            
            pyplot.title(Well_ID + ": "+proj_name, size=17, fontweight = 'bold')
            pyplot.ylabel('GWL [m asl]', size=15)
            pyplot.xlabel('Date',size=15)
            pyplot.legend(fontsize=15,fancybox = False, framealpha = 1, edgecolor = 'k')
            pyplot.tight_layout()
            pyplot.grid(b=True, which='major', color='#666666', alpha = 0.3, linestyle='-')
            pyplot.xticks(fontsize=14)
            pyplot.yticks(fontsize=14)

            pyplot.savefig('./'+Well_ID+'/Climproj_'+Well_ID+"_"+proj_name+'_CNN_PT.png', dpi=600)
            pyplot.show()
        
            #print sim data
            printdf = pd.DataFrame(data=sim_members,index=data.index)
            printdf.to_csv('./'+Well_ID+'/ensemble_member_values_CNN_'+Well_ID+'_'+proj_name+'.txt',sep=';')
            
            printdf = pd.DataFrame(data=sim_members_uncertainty,index=data.index)
            printdf.to_csv('./'+Well_ID+'/ensemble_member_errors_CNN_'+Well_ID+'_'+proj_name+'.txt',sep=';')
            
            printdf = pd.DataFrame(data=np.c_[sim_mean,y_err],index=data.index)
            printdf = printdf.rename(columns={0: 'Sim', 1:'Sim_Error'})
            printdf.to_csv('./'+Well_ID+'/ensemble_mean_values_CNN_'+Well_ID+'_'+proj_name+'.txt',sep=';', float_format = '%.6f')
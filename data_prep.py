import pandas as pd
import constants as c

def get_data(path):
    
    # here we do all the necessary data preparations for processing (data cleanup)
    data = pd.read_csv(path, engine='pyarrow', dtype_backend='pyarrow')
    data = data.dropna()
    data['CarName'] = data['CarName'].astype("string").str.split(" ").str.get(0) #Trim car model due to simplicity of the program
    data = data.drop_duplicates() # duplicates are caused due to car family trim

    return data

def get_inputs(data):
    
    # prepare feature set
    input_data = data[c.DATA_FEATURE_COLS]

    # one hot encoding feature set
    input_data = pd.get_dummies(input_data, columns=c.DATA_DUMMY_COLS, drop_first=False, dtype='int')

    return input_data

def get_targets(data):
    
    target_data = data[c.DATA_TARGET_COLS]

    return target_data





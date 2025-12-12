import pandas as pd
import constants as c

def get_data(path):
    
    data = pd.read_csv(path, engine='pyarrow', dtype_backend='pyarrow')

    # here we do all the necessary data preparations for processing (data cleanup)
    data = data[c.DATA_RELEVANT_COLS] # work only with columns we care about
    data['CarName'] = data['CarName'].astype("string").str.split(" ").str.get(0) #trim car model due to simplicity of the program AUDI A4 -> AUDI
    data.dropna(inplace=True) #drop rows that contain empy values

    # here perform the specific input data operations -> for 2 same feature sets we get 2 different prices due to CarName trim -> exclude those rows
    input_data = data.drop(columns=c.DATA_TARGET_COLS)  #remove price from the equation so the diplicates are thrown out
    target_data = data.drop(columns=c.DATA_FEATURE_COLS)  #remove price from the equation so the diplicates are thrown out
    input_data = pd.get_dummies(input_data, columns=c.DATA_DUMMY_COLS, drop_first=False, dtype='int') # one hot encoding feature set
    input_data.drop_duplicates(input_data, inplace=True) # duplicates are caused due to car family trim -> drop aftergetting dummies

    data = pd.concat([input_data, target_data], axis=1, join='inner') #concatenate the dataframes back together on row axis, based on predefined sorting indexes

    return data, input_data, target_data


def get_inputs(data):
    return data.drop

def get_targets(data):   
    target_data = data[c.DATA_TARGET_COLS]
    return target_data





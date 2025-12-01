import pandas as pd

def get_inputs():
  
  car_data_df = pd.read_csv('./archive/CarPrice_Assignment.csv', engine='pyarrow', dtype_backend='pyarrow')
  
  car_data_df = car_data_df.dropna()

  # prepare feature set
  feature_set = car_data_df
  feature_set['CarName'] = feature_set['CarName'].astype("string").str.split(" ").str.get(0)
  feature_columns = ['CarName', 'fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'wheelbase', 'carlength', 'carwidth', 'curbweight', 'enginetype', 'cylindernumber', 'enginesize', 'horsepower', 'citympg', 'highwaympg']

  input_data = feature_set[feature_columns]


  # one hot encoding feature set
  input_data = pd.get_dummies(input_data, columns=['CarName', 'fueltype', 'carbody', 'drivewheel', 'aspiration', 'enginelocation', 'enginetype', 'cylindernumber'], drop_first=False, dtype='int')

  return input_data




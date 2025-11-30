import pandas as pd

car_data_df = pd.read_csv('./archive/CarPrice_Assignment.csv', engine='pyarrow', dtype_backend='pyarrow')

car_data_df = car_data_df.dropna()

print(car_data_df.info())



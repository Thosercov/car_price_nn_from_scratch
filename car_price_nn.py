import pandas as pd
import numpy as np
import data_prep as dp

input_data = dp.get_input()

input_data = input_data.convert_dtypes()

boolean_columns = input_data.select_dtypes(include='bool').columns
input_data[boolean_columns] = input_data[boolean_columns].astype(int)

print(input_data)





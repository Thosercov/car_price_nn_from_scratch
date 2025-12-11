import numpy as np
import pandas as pd

class Standardization_Z_score:     

    def standardize(self, inputs):

        is_column_dummy_series = np.all((inputs == 0) | (inputs == 1), axis=0) # returns pandas boolean series - all values are (X == 0 or X == 1)
        dummy_columns = []
        num_columns = []
        for column_name, is_dummy in is_column_dummy_series.items():
            if is_dummy:
                dummy_columns.append(column_name)
            else:
                num_columns.append(column_name)

        #create 2 separate dataframes
        inputs_numerical = inputs.drop(columns=dummy_columns)
        inputs_dummies = inputs.drop(columns=num_columns)

        self.standard_deviation = np.array(np.std(inputs_numerical, axis=0).values) #population standard deviation
        self.mean = np.array(inputs_numerical.mean(axis=0)) 

        inputs_standardized = (inputs_numerical - self.mean) / self.standard_deviation # z-score standardization formula 

        return pd.concat([inputs_standardized, inputs_dummies], axis=1) #concatenate the dataframes back together on row axis, based on predefined sorting indexes



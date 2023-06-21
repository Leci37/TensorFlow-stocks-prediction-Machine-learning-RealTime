import numpy as np
import pandas as pd
from typing import List

class Diff:
    '''
    Log 1st Order Difference for Data Preprocessing
    Each Dataset should has its own LogDiff Class.
    - i.e., a LogDiff for Training Data; another LogDiff for Testing Data.
    '''
    def __init__(self):
        self.memory = {}
        
    def transform(self, x: pd.DataFrame, columns: List[str]):
        temp = {}
        y = x.copy()
        for column in columns:
            # Record the first value of the sequence
            self.memory[column] = y[column].iloc[0]
            temp[column] = y[column] - y[column].shift(1)

        # Remove the first NaN data from the input data
        output = x.copy()[1:]
        for column in columns:
            output[column] = temp[column]

        # print(output)
        return output


    def inverse_transform(self, x: pd.DataFrame, columns: List[str]):
        y = x.copy().shift(1, axis=0)
        for column in columns:
            # Set back the original value
            y[column].iloc[0] = np.log(self.memory[column])
            y[column] = np.exp(np.cumsum(y[column]))
        return y[1:]
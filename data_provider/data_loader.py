### THIS IS BOILER PLATE FOR DATA LOADER CREATION
import os 
import pandas as pd 
import numpy as np 
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import warnings 

warnings.filterwarnings('ignore')

class Dataset_A(Dataset):
    def __init__(self, root_path, flag = 'train'):
        self.root_path = root_path
        self.flag = flag

    def __read_data__(self):
        # HOW TO READ THE DATA

        self.data_x = np.ones((1,1))
        self.data_y = np.ones((1,1))

    
    def __getitem__(self, index):
        # RETURN SAMPLE OF DATASET
        s_begin = index 
        s_end = s_begin

        seq_x = self.data_x[0]
        seq_y = self.data_y[0]

        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x)
    

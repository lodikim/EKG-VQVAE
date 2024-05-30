import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

#######################################################################################
## Dataset Code
#######################################################################################

class Dataset_EKG(Dataset):
    def __init__(self, root_path = '/home/bryanswkim/mamba/ekg-vqvae/', size=None, flag='train',
                 data_path=None, scale=True):

        if size == None:
            self.seq_len = 1024
            self.pred_len = 1024
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        #self.dataset_len = dataset_len

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        # Get all ekg files
        ekg_files = []
        for file in os.listdir(self.root_path):
            if file.startswith('ekg') and file.endswith('.csv'):
                ekg_files.append(os.path.join(self.root_path, file))

        # Iterate through each ekg file
        data_x = []
        data_y = []
        for file in ekg_files:
            df_raw = pd.read_csv(file)

            cols = list(df_raw.columns)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols]
            df_raw = df_raw[:]

            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]

            if self.scale:
                train_data = df_data
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            data = np.float32(data)
            
            data_x.append(data)
            data_y.append(data)


        self.data_x = np.concatenate(data_x)
        self.data_y = np.concatenate(data_y)
        
    def __getitem__(self, index):
        x_begin = index
        x_end = x_begin + self.seq_len
        y_begin = x_end
        y_end = y_begin + self.pred_len

        seq_x = self.data_x[x_begin:x_end]
        seq_y = self.data_y[y_begin:y_end]
        
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class NoOverlap_Dataset_EKG(Dataset):
    def __init__(self, root_path = '/home/bryanswkim/mamba/ekg-vqvae/', size=None, flag='train',
                 data_path='example_ekg.csv',
                 scale=True):

        if size == None:
            self.seq_len = 1024
            self.pred_len = 1024
        else:
            self.seq_len = size[0]
            self.pred_len = size[1]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        '''
        border_start_all = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len] # [0, 0.7D - L, 0.8D - L]
        border_ended_all = [num_train, num_train + num_vali, len(df_raw)] # [0.7D, 0.8D, 1.0D]
        
        border_start = border_start_all[self.set_type]
        border_ended = border_ended_all[self.set_type]
        '''

        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        if self.scale: # Scaling with sklearn preprocessing
            #train_data = df_data[border_start_all[0]:border_ended_all[0]]
            train_data = df_data
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        data = np.float32(data)

        chunked_data_x = []
        chunked_data_y = []
        chunk_num = int(len(data)/self.seq_len)     # 16816

        num_train = int(chunk_num * 0.7)
        num_test = int(chunk_num * 0.2)
        num_vali = chunk_num - num_train - num_test

        for i in range(chunk_num-1):       # len(data) / self.seq_len
            chunk_start = i*self.seq_len
            chunked_data_x.append(data[chunk_start:chunk_start+self.seq_len])
            chunked_data_y.append(data[chunk_start+self.seq_len:chunk_start+2*self.seq_len])

        self.data_x = chunked_data_x
        self.data_y = chunked_data_y
        
    def __getitem__(self, index):

        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        
        return seq_x, seq_y

    def __len__(self):
        #return len(self.data_x) - self.seq_len - self.pred_len + 1
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

'''
CodeRow = namedtuple('CodeRow', ['top', 'bottom', 'filename'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename


class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom), row.filename
'''
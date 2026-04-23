import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
import lmdb
import pickle

def to_tensor(array):
    return torch.from_numpy(array).float()

class CustomDataset(Dataset):
    def __init__(self, data_dir, mode='train'):
        super(CustomDataset, self).__init__()
        self.data_dir = data_dir  # Save the path for later
        self.mode = mode
        self.db = None            # Do NOT open it permanently yet
        
        # 1. Open temporarily just to get the keys
        temp_db = lmdb.open(data_dir, readonly=True, lock=False, readahead=True, meminit=False)
        with temp_db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get('__keys__'.encode()))[mode]
            
        # 2. Close it immediately so val_set can use the folder!
        temp_db.close() 

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        # 3. LAZY INITIALIZATION: Open the DB the first time a batch is requested.
        # This makes it safe for both 'val' and 'train', AND safe for num_workers=16!
        if self.db is None:
            self.db = lmdb.open(self.data_dir, readonly=True, lock=False, readahead=True, meminit=False)
            
        key = self.keys[idx]
        with self.db.begin(write=False) as txn:
            pair = pickle.loads(txn.get(key.encode()))
            
        data = pair['sample']
        label = pair['label']
        
        return data / 100.0, label

    def collate(self, batch):
        x_data = np.array([x[0] for x in batch])
        y_label = np.array([x[1] for x in batch])
        return to_tensor(x_data), to_tensor(y_label)


class LoadDataset(object):
    def __init__(self, params):
        self.params = params
        self.dataset_dir = params.dataset_dir	# You need to pass the FOLDER that contains the lock.mdb and the data.mdb files.

    def get_data_loader(self):
        train_set = CustomDataset(self.dataset_dir, mode='train')
        val_set = CustomDataset(self.dataset_dir, mode='val')
        test_set = CustomDataset(self.dataset_dir, mode='test')
        print(len(train_set), len(val_set), len(test_set))
        data_loader = {
            'train': DataLoader(
                train_set,
                batch_size=self.params.batch_size,
                collate_fn=train_set.collate,
                shuffle=True,
                pin_memory=True,
                num_workers=12,
            ),
            'val': DataLoader(
                val_set,
                batch_size=self.params.batch_size,
                collate_fn=val_set.collate,
                shuffle=False,
                pin_memory=True,
                num_workers=12,
            ),
            'test': DataLoader(
                test_set,
                batch_size=self.params.batch_size,
                collate_fn=test_set.collate,
                shuffle=False,
                pin_memory=True,
                num_workers=12,
            ),
        }
        return data_loader

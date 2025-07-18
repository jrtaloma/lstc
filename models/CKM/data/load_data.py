import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader


def standardize(seq):
    return (seq - seq.mean()) / seq.std()


class Dataset_UCR():
    def __init__(self, dataset_name):
        self.dir_dataset = os.path.join('./datasets', 'UCRArchive_2018', dataset_name)
        train_set = pd.read_csv(os.path.join(self.dir_dataset, f'{dataset_name}_TRAIN.tsv'), sep='\t').values
        test_set = pd.read_csv(os.path.join(self.dir_dataset, f'{dataset_name}_TEST.tsv'), sep='\t').values
        data = np.concatenate([train_set, test_set], axis=0)
        X = data[:, 1:]
        for i in range(len(X)):
            X[i] = standardize(X[i])
        y = data[:, 0]
        y = LabelEncoder().fit_transform(y)  # sometimes y are strings or start from 1
        assert y.min() == 0  # assert y are integers and start from 0
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int32)
        self.n_clusters = len(np.unique(self.y))


class CustomDataset(Dataset):
    def __init__(self, time_series, labels):
        """
        This class creates a torch dataset.
        """
        self.time_series = time_series
        self.labels = labels
        self.n_clusters = len(np.unique(self.labels))
        self.n_input = self.time_series.shape[1]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        time_series = torch.tensor(self.time_series[idx])
        label = torch.tensor(self.labels[idx])
        idx_ = torch.tensor(np.array(idx))
        return (time_series, label, idx_)


def get_loader(dataset_name, batch_size, shuffle, drop_last, num_workers):
    dataset_ucr = Dataset_UCR(dataset_name)
    dataset = CustomDataset(dataset_ucr.X, dataset_ucr.y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataset, loader

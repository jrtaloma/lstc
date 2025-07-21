import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from .augmentation import rotation, permutation, time_warp


def standardize(seq):
    return (seq - seq.mean()) / seq.std()


def normalize(seq):
    return (seq - seq.min()) / (seq.max() - seq.min())


class Dataset_UCR():
    def __init__(self, dataset_name):
        dir_dataset = os.path.join('./datasets', 'UCRArchive_2018', dataset_name)
        train_set = pd.read_csv(os.path.join(dir_dataset, f'{dataset_name}_TRAIN.tsv'), sep='\t').values
        test_set = pd.read_csv(os.path.join(dir_dataset, f'{dataset_name}_TEST.tsv'), sep='\t').values
        data = np.concatenate([train_set, test_set], axis=0)
        X = data[:, 1:]
        for i in range(len(X)):
            X[i] = standardize(X[i])
        X_a = time_warp(permutation(rotation(np.expand_dims(X, axis=2)))).squeeze(2)
        y = data[:, 0]
        y = LabelEncoder().fit_transform(y)  # sometimes y are strings or start from 1
        assert y.min() == 0  # assert y are integers and start from 0
        self.X = X.astype(np.float32)
        self.X_a = X_a.astype(np.float32)
        self.y = y.astype(np.int32)
        self.n_clusters = len(np.unique(self.y))


class Dataset_M5():
    def __init__(self, dataset_name='m5-forecasting-accuracy'):
        dir_dataset = os.path.join('./datasets', dataset_name)
        sales = np.load(os.path.join(dir_dataset, 'sales.npy'), allow_pickle=True).astype(np.float32)
        labels = np.load(os.path.join(dir_dataset, 'cat_store_id.npy'), allow_pickle=True)
        labels = LabelEncoder().fit_transform(labels)
        assert labels.min() == 0  # assert labels are integers and start from 0
        for i in range(len(sales)):
            sales[i] = normalize(sales[i])
        sales_a = time_warp(permutation(rotation(np.expand_dims(sales, axis=2)))).squeeze(2)
        self.X = sales
        self.X_a = sales_a.astype(np.float32)
        self.y = labels
        self.n_clusters = len(np.unique(self.y))


class CustomDataset(Dataset):
    def __init__(self, time_series, time_series_augmented, labels):
        """
        This class creates a torch dataset.
        """
        self.n_input = time_series.shape[1]
        self.new_seq_len = self._compute_divisible_len(self.n_input)
        self.time_series = time_series[:, :self.new_seq_len]
        self.time_series_augmented = time_series_augmented[:, :self.new_seq_len]
        self.labels = labels
        self.n_clusters = len(np.unique(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        time_series = torch.from_numpy(self.time_series[idx])
        time_series_augmented = torch.from_numpy(self.time_series_augmented[idx])
        label = torch.tensor(self.labels[idx])
        return time_series, time_series_augmented, label

    def _compute_divisible_len(self, n_input):
        size = (n_input // 32) * 32
        while size > 0:
            if size % 12 == 0:
                break
            size -= 32
        return size


def get_loader(dataset_name, batch_size, shuffle, drop_last, num_workers):
    if dataset_name != 'm5-forecasting-accuracy':
        dataset_ucr = Dataset_UCR(dataset_name)
        dataset = CustomDataset(dataset_ucr.X, dataset_ucr.X_a, dataset_ucr.y)
    else:
        dataset_m5 = Dataset_M5(dataset_name)
        dataset = CustomDataset(dataset_m5.X, dataset_m5.X_a, dataset_m5.y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=drop_last, num_workers=num_workers)
    return dataset, loader
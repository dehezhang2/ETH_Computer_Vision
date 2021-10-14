import gzip

import numpy as np

import os

import torch
from torch.utils.data import Dataset


IMAGE_SIZE = 28


# NNIST reading code from https://stackoverflow.com/a/53570674.
def load_images(data_path):
    f = gzip.open(data_path,'r')

    f.read(16)
    buf = f.read()
    data = np.frombuffer(buf, dtype=np.uint8)
    data = data.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)

    return data


def load_labels(label_path):
    f = gzip.open(label_path,'r')
    
    f.read(8)
    buf = f.read()
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    
    return labels


class MNISTDataset(Dataset):
    def __init__(self, split='train'):
        super().__init__()
        assert split in ['train', 'test'], f'Split parameters "{split}" must be either "train" or "test".'
        # Set up paths for either train or test data from disk based on split parameter.
        # Data is located in the folder "data".
        # Save samples and annotations to a class member.
        data_path = os.path.join('data', f'{split}-images-idx3-ubyte.gz')
        labels_path = os.path.join('data', f'{split}-labels-idx1-ubyte.gz')
        self.data = load_images(data_path)
        self.annotations = load_labels(labels_path)
            
    def __len__(self):
        # Returns the number of samples in the dataset.
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        # Returns the sample with index idx from the dataset as a torch tensor.
        sample = self.data[idx]
        annotation = self.annotations[idx]

        # Images are generally represented as uint8 matrices ([0 .. 255]).
        # Normalize the data between -1 and 1!
        raise NotImplementedError()
        sample = normalize(sample)
        
        return {
            'input': torch.from_numpy(sample).float(),
            'annotation': torch.tensor(annotation).long()
        }


def normalize(sample):
    raise NotImplementedError()
    new_sample = None
    return new_sample

import pickle
import torch
from torch.utils.data import Dataset

class TimeSeriesInferenceBatchDataset(Dataset):
    def __init__(self, path):
        super().__init__()

        self.path = path
        with open('pickle.dat', 'rb') as file:
            # Load the object from the file
            self.data = pickle.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "X": torch.Tensor(self.data[0][idx]), 
            "y": torch.Tensor(self.data[1][idx])
        }
import torch 
from torch.utils.data import Dataset 

import numpy as np 

class DigitDataset(torch.utils.data.Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        label = row.label
        img_vec = np.array(row[1:])
        image = np.reshape(img_vec, (28,28,1)) / 255.0

        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)

        return {
            'images' : image, 
            'labels' : label
        }
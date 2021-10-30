from config import * 
from dataset import * 
from model import * 
from engine import *

import pandas as pd
import os 

from sklearn.model_selection import train_test_split


def run():

    df = pd.read_csv(TRAIN_CSV)
    df_train, df_valid = train_test_split(df, test_size=0.2, random_state=42)

    trainset = DigitDataset(df_train)
    validset = DigitDataset(df_valid)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        shuffle = True, 
        batch_size = BATCH_SIZE
    )

    validloader = torch.utils.data.DataLoader(
        validset,
        batch_size = BATCH_SIZE
    )

    model = DigitModel()
    optimizer = torch.optim.Adam(model.parameters(), lr = LR)

    best_valid_loss = np.Inf

    for i in range(EPOCHS):
        train_loss = train_fn(model,trainloader,optimizer,i+1)
        valid_loss = eval_fn(model,validloader,i+1)

        if valid_loss < best_valid_loss:
            torch.save(model.state_dict(), 'best-digit-model.pt')
            best_valid_loss = valid_loss

if __name__ == '__main__':
    run() 

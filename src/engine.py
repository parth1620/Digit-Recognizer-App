import torch 

from tqdm import tqdm 
from sklearn.metrics import accuracy_score
from torch import nn 

def get_acc(logits, labels):
    preds = torch.argmax(logits, dim = 1).detach().numpy()
    labels = labels.detach().numpy()
    return accuracy_score(preds, labels)

def train_fn(model, dataloader, optimizer, current_epoch):

    model.train() #on-dropout, batchnorm 
    total_loss = 0.0
    total_acc = 0.0
    l_loss = 0.0
    l_acc = 0.0

    tk = tqdm(dataloader, desc = "Epoch" + " [TRAIN] " + str(current_epoch))

    for t,data in enumerate(tk):
        logits, loss = model(**data)

        optimizer.zero_grad()
        loss.backward() #dw, db
        optimizer.step()

        total_loss+=loss.item()
        l_loss = total_loss / (t+1)

        total_acc+=get_acc(logits, data['labels'])
        l_acc = total_acc / (t+1)

        tk.set_postfix({'loss' : '%.6f' %float(l_loss), 'acc' : '%.6f' %float(l_acc)})

    total_loss = total_loss / len(dataloader)
    total_acc = total_acc / len(dataloader)  

    return total_loss

def eval_fn(model, dataloader, current_epoch):

    model.eval() #on-dropout, batchnorm 
    total_loss = 0.0
    total_acc = 0.0
    l_loss = 0.0
    l_acc = 0.0

    tk = tqdm(dataloader, desc = "Epoch" + " [VALID] " + str(current_epoch))

    for t,data in enumerate(tk):
        
        logits, loss = model(**data)

        total_loss+=loss.item()
        l_loss = total_loss / (t+1)

        total_acc+=get_acc(logits, data['labels'])
        l_acc = total_acc / (t+1)

        tk.set_postfix({'loss' : '%.6f' %float(l_loss), 'acc' : '%.6f' %float(l_acc)})

    total_loss = total_loss / len(dataloader)
    total_acc = total_acc / len(dataloader)

    return total_loss

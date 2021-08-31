from __future__ import print_function, division

import time
import copy
from tqdm import tqdm
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler


from model import *
from data import *
from metric import *
import argparse

import warnings
warnings.simplefilter("ignore", UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--model',          type=str,   default = 'resnet18')
parser.add_argument('--save_folder',    type=str,   default = './weights/setup')
parser.add_argument('--trained_model',  type=str,   default = 'none')
parser.add_argument('--bs',             type=int,   default = 128)
parser.add_argument('--epoch',          type=int,   default = 25)
parser.add_argument('--metric',         type=str,   default = 'arcface')
parser.add_argument('--lr',         type=float,   default = 0.001)
parser.add_argument('--s',         type=int,   default = 30)
args = parser.parse_args()


train_dataset = MS1MDataset()
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

num_classes = 86876

if args.metric == 'arcface':
  metric = ArcMarginProduct(512, num_classes, s=args.s, m=0.5, easy_margin=False)
elif args.metric == 'vsoftmax':
  metric = VirtualSoftmax(512, num_classes, train=True)

if args.model == 'resnet18':
  model = resnet18()
elif args.model == 'resnet50':
  model = resnet50()
elif args.model == 'resnet101':
  model = resnet101()
else:
  print('model is not selected')

  
if args.trained_model != 'none':
  load_model_from = './weights/setup/' + args.trained_model + '.pth'
  print('load model from', load_model_from, '\n')
  model.load_state_dict(torch.load(load_model_from))

metric = metric.to(device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = [{'params': model.parameters()},
                                 {'params': metric.parameters()}], 
                       lr=args.lr) #, weight_decay=5e-4)

scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)


def train_model(model, metric, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    model.train()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in tqdm(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            features = model(inputs) 
            outputs = metric(features, labels)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        print(running_corrects.item(), ' corrected')

        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = running_corrects.double() / len(dataloader)

        print('Loss: {:.4f} | Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), args.save_folder+'/acc_'+str(int(epoch_acc * 100))+'.pth')


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best train Acc: {:4f}'.format(best_acc))

    torch.save(model.state_dict(), args.save_folder+ '/' + args.model + str(args.s) + '.pth')
    return model


if __name__ == "__main__":
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)

    model = train_model(model, metric, criterion, optimizer, scheduler, args.epoch)
    
    
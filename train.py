from __future__ import print_function, division
import time
import copy
from tqdm import tqdm
import os
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from utils import validation
from model import *
from data import *
from metric import *
import argparse

from torchvision import datasets
from torchvision import transforms

import warnings
warnings.simplefilter("ignore", UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--model',          type=str,   default = 'resnet18')
parser.add_argument('--metric',         type=str,   default = 'arcface')
parser.add_argument('--trained_model',  type=str,   default = 'none')
parser.add_argument('--bs',             type=int,   default = 128)
parser.add_argument('--epoch',          type=int,   default = 25)
parser.add_argument('--lr',         type=float,   default = 0.001)
parser.add_argument('--s',         type=int,   default = 30)
args = parser.parse_args()

num_classes = 86876

train_dataset = MS1MDataset()
# train_dataset = datasets.ImageFolder(
#     '/content/content/faces_glintasia/face_asian',
#     transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
# )

if args.model == 'r18': 
    model = resnet18()     
elif args.model == 'r50':
    model = resnet50()
elif args.model == 'r101':
    model = resnet101() 

metric = ArcMarginProduct(512, num_classes, s=args.s, m=0.5, easy_margin=False)
dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


if args.trained_model != 'none':
  load_model_from = './weights/setup/' + args.trained_model + '.pth'
  print('load model from', load_model_from, '\n')
  model.load_state_dict(torch.load(load_model_from, map_location=torch.device('cpu')))
  # model.load_state_dict(torch.load(load_model_from))

metric = metric.to(device)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(params = [{'params': model.parameters()},
                                 {'params': metric.parameters()}], 
                       lr=args.lr) #, weight_decay=5e-4)

scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.6)


def train_model(model, metric, criterion, optimizer, scheduler, num_epochs):
    
    val_sets = [
        # 'ms1m',
        # 'lfw',
        # 'cfp_ff',
        # 'megaface',

        # 'cfp_fp', 
        'cplfw',  
        # 'vgg2_fp',

        # 'asian'
    ]
    auc_scores = pd.DataFrame(columns=val_sets)

    for epoch in range(1):

        iters = 1
        for inputs, labels in tqdm(dataloader):
            model.train()
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()

            features = model(inputs) 
            outputs = metric(features, labels)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            
            if iters % 2000 == 0:
                score = pd.Series(validation(model, device, val_sets), index=val_sets)
                auc_scores = auc_scores.append(score, ignore_index=True)
                print('\n', score)
            iters += 1
        
        scheduler.step()

    fname = args.model + '-s' + str(args.s) + '-e' + str(args.epoch) + '-lr' + str(args.lr).replace('.', '')
    auc_scores.to_csv('./train_eval/' + fname + '.csv', index=False)
    torch.save(model.state_dict(), './weights/setup/' + fname + '.pth')
    return model


if __name__ == "__main__":
    model = train_model(model, metric, criterion, optimizer, scheduler, args.epoch)
    
    
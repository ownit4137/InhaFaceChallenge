import pandas as pd
import numpy as np
import os
import random
from PIL import Image
from torchvision import transforms
import torch
import argparse

import sys
sys.path.append('/content/drive/MyDrive/FACE_AI_share/model')
import resnet

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--filename', type=str, default = 'new')
  parser.add_argument('--size', type=int, default = 6000)
  args = parser.parse_args()

  with open('/content/inha_data/ID_List.txt', 'r') as f:
    train_id = f.readlines()

  train_id = [id.strip() for id in train_id]
  train_id = train_id[1:500000+1]

  classes=[]
  face_images_path=[]
  for i in range(len(train_id)):
    classes.append(int(train_id[i].split()[0]))
    face_images_path.append(train_id[i].split()[1])

  left_index = random.sample([i for i in range(500000)], args.size)

  right_index = []
  for i in range(args.size):
    if i % 2 == 0:
      right_index.append(left_index[i]+1)
    else:
      right_index.append(random.randint(1, 500000))

  left_paths = []
  right_paths = []

  for i in range(args.size):
    left_paths.append(face_images_path[left_index[i]][:-4])
    right_paths.append(face_images_path[right_index[i]][:-4])

  new_trainset = pd.DataFrame({'left_paths': left_paths, 'right_paths': right_paths, 'left_index': left_index, 'right_index': right_index})

  def make_answer(left, right):
    if left.split('/')[1] == right.split('/')[1]:
      return 1
    else:
      return 0

  new_trainset['face_images'] = new_trainset['left_paths'] + ' ' + new_trainset['right_paths']
  new_trainset['answer'] = new_trainset.apply(lambda x: make_answer(x['left_paths'], x['right_paths']), axis=1)
  new_trainset.drop(labels=['left_paths', 'right_paths', 'left_index', 'right_index'], axis=1, inplace=True)

  new_trainset.to_csv('./' + args.filename + '.csv', index=False)
  print(args.filename + '.csv created')
import os
import shutil

# moving validation images

root = '/content/kfacesample'

def fileout(path):
    global root
    if os.path.isfile(path) == True:
        filename = os.path.split(path)[1]
        shutil.move(path, os.path.join(root, filename))
    else:
        dirs = os.listdir(path)
        for d in dirs:
            fileout(os.path.join(path, d))

fileout(root)
!rm -rf /content/kfacesample/S*


dirs = os.listdir(root)
for i, fname in enumerate(dirs):
    origin = os.path.join(root, fname)
    to = os.path.join(root, str(i) + '.jpg')
    os.rename(origin, to)


# crop face from image

from tqdm import tqdm
import cv2
import face_recognition
from google.colab.patches import cv2_imshow

count = 0

for i in range(10800):
  im = cv2.imread("./kfacesample/" + str(i) + ".jpg")
  image = face_recognition.load_image_file("./kfacesample/" + str(i) + ".jpg")
  face_locations = face_recognition.face_locations(image, model="cnn")

  
  for (t, r, b, l) in face_locations: 
    center = [int((l + r) / 2)+ 20 , int((t + b)/ 2) + 20]

    img = cv2.copyMakeBorder(
    im, 20, 20, 20, 20,
    borderType=cv2.BORDER_CONSTANT,
    value=[0, 0, 0]
    )

    cropped_img = img[center[1] - 56:center[1] + 56, center[0] - 56:center[0] + 56]

    cv2.imwrite('./kface/' + str(count) + '.jpg', cropped_img)
    count += 1

# transform faces_glintasia

from mxnet import recordio
import mxnet as mx
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
import os
path_imgidx = '/content/faces_glintasia/train.idx'
path_imgrec = '/content/faces_glintasia/train.rec'
root_path = '/content/faces_glintasia/face_asian'
imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

last = 0
cnt = 0
for i in tqdm(range(2830146)):
    header, s = recordio.unpack(imgrec.read_idx(i+1))
    img = mx.image.imdecode(s).asnumpy()

    dst = os.path.join(root_path,str(int(header.label[0])))
    if not os.path.exists(dst):
        os.makedirs(dst)
        last = int(header.label[0])
        cnt=0
    cv2.imwrite(os.path.join(dst,f'{cnt}.jpg'),cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    cnt+=1


# make validation set randomly

import os
import shutil
import random
import pandas as pd
from tqdm import tqdm

df = pd.DataFrame()

left_file_list = []
right_file_list = []
answer = []

kroot = '/content/valid_kface/valid'
aroot = '/content/content/faces_glintasia/face_asian'
kdirs = len(os.listdir(kroot))
adirs = len(os.listdir(aroot))

# 1s
for i in range(2500):
    dirs = len(os.listdir(kroot))
    left_file_list.append(os.path.join(kroot, str(random.randint(0, dirs - 1)) + '.jpg'))
    right_file_list.append(os.path.join(kroot, str(random.randint(0, dirs - 1)) + '.jpg'))
    answer.append(1)

# 0s
for i in range(2500):
    left_file_list.append(os.path.join(root, str(random.randint(0, kdirs - 1)) + '.jpg'))
    
    afiles = 0
    while afiles < 1:
        asubdir = os.path.join(aroot, str(random.randint(0, adirs - 1)))
        afiles = len(os.listdir(asubdir))
    right_file_list.append(os.path.join(asubdir, str(random.randint(0, afiles - 1)) + '.jpg'))
    answer.append(0)

for i in range(1000):
    if i % 2 == 0:
        asubdir = ''
        afiles = 0
        while afiles < 1:
            asubdir = os.path.join(aroot, str(random.randint(0, adirs - 1)))
            afiles = len(os.listdir(asubdir))
        left_file_list.append(os.path.join(asubdir, str(random.randint(0, afiles - 1)) + '.jpg'))
        right_file_list.append(os.path.join(asubdir, str(random.randint(0, afiles - 1)) + '.jpg'))
        answer.append(1)
    else:
        left_idx = 0
        afiles = 0

        while afiles < 1:
            left_idx = random.randint(0, adirs - 1)
            
            asubdir = os.path.join(aroot, str(left_idx))
            afiles = len(os.listdir(asubdir))
        left_file_list.append(os.path.join(asubdir, str(random.randint(0, afiles - 1)) + '.jpg'))

        afiles = 0
        right_idx = left_idx
        while afiles < 1 or left_idx == right_idx:
            right_idx = random.randint(0, adirs - 1)
            
            asubdir = os.path.join(aroot, str(right_idx))
            afiles = len(os.listdir(asubdir))
        right_file_list.append(os.path.join(asubdir, str(random.randint(0, afiles - 1)) + '.jpg'))
        answer.append(0)

        if left_idx == right_idx:
            print('aa')


df['left'] = left_file_list
df['right'] = right_file_list
df['answer'] = answer

df.to_csv('/content/answer.csv')
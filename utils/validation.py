import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import os
from sklearn.metrics import roc_auc_score
from model import *
from tqdm import tqdm

data_transform = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def cos_sim(a, b):
    return F.cosine_similarity(a, b)

def validation(model, device, val_list):
    scores = []
    for name in val_list:
        scores.append(do_validation(model, device, name))
    return scores
    
def do_validation(model, device, name):
    if name=='ms1m':
        ans_path = '/content/drive/MyDrive/FACE_AI_share/train_eval/ms1m_valid1.csv'
        valid_root = '/content/inha_data/'
    else:
        ans_path = '/content/valid_' + name + '/answer.csv'
        valid_root = '/content/valid_' + name + '/valid/'

    model = model.to(device)
    model.eval()

    submission = pd.read_csv(ans_path)
    size = len(submission)
    batch_size = int(size / 40) # mod

    left_test_paths  = list()
    right_test_paths = list()

    for i in range(len(submission)):
        left_test_paths.append(submission['face_images'][i].split()[0])
        right_test_paths.append(submission['face_images'][i].split()[1])

    left_test = list()
    for left_test_path in left_test_paths:
        img = Image.open(valid_root + left_test_path + '.jpg').convert("RGB")
        img = data_transform(img)
        left_test.append(img) 
    left_test = torch.stack(left_test)

    left_infer_result_list = list()
    with torch.no_grad():
        for i in range(0, size//batch_size):      # remv
            i = i * batch_size
            tmp_left_input = left_test[i:i+batch_size]
            left_infer_result = model(tmp_left_input.to(device))
            left_infer_result_list.append(left_infer_result)

        left_infer_result_list = torch.stack(left_infer_result_list, dim=0).view(-1, 512)

    right_test = list()
    for right_test_path in right_test_paths:
        img = Image.open(valid_root + right_test_path + '.jpg').convert("RGB") 
        img = data_transform(img)
        right_test.append(img)
    right_test = torch.stack(right_test)

    right_infer_result_list = list()
    with torch.no_grad():
        for i in range(0, size//batch_size):
            i = i * batch_size
            tmp_right_input = right_test[i:i+batch_size]
            right_infer_result = model(tmp_right_input.to(device))
            right_infer_result_list.append(right_infer_result)

        right_infer_result_list = torch.stack(right_infer_result_list, dim=0).view(-1, 512)

    cosin_similarity = cos_sim(left_infer_result_list, right_infer_result_list)
    submission['pred'] = cosin_similarity.tolist() 
    auc_score = roc_auc_score(submission['answer'], submission['pred'])

    return auc_score
import pandas as pd
from model import *
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse

import warnings
warnings.simplefilter("ignore", UserWarning)

data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def cos_sim(a, b):
    return F.cosine_similarity(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained_model',  type=str,   default = 'final.pth')
    parser.add_argument('--bs',             type=int,   default = 60)
    parser.add_argument('--model',          type=str,   default = 'resnet18')
    parser.add_argument('--src',     type=str,   default = '/content/inha_data/sample_submission.csv')
    args = parser.parse_args()

    load_model_from = './weights/setup/' + args.trained_model + '.pth'
    save_to = '/content/drive/MyDrive/FACE_AI_share/submissions/' + args.trained_model + '.csv'
    print('load model from', load_model_from)
    print('save submission file to', save_to, '\n')

    submission = pd.read_csv(args.src)

    if args.model == 'resnet18':
      model = resnet18()
    elif args.model == 'resnet50':
      model = resnet50()
    elif args.model == 'resnet101':
      model = resnet101()
    else:
      print('model is not selected')

    model.load_state_dict(torch.load(load_model_from))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    left_test_paths  = list()
    right_test_paths = list()


    for i in range(len(submission)):
        left_test_paths.append(submission['face_images'][i].split()[0])
        right_test_paths.append(submission['face_images'][i].split()[1])

    left_test = list()
    for left_test_path in left_test_paths:
        img = Image.open("/content/inha_data/test/" + left_test_path + '.jpg').convert("RGB")
        img = data_transform(img)
        left_test.append(img) 
    left_test = torch.stack(left_test)

    left_infer_result_list = list()
    with torch.no_grad():
        batch_size = args.bs
        for i in tqdm(range(0, 6000//args.bs)):
            i = i * batch_size
            tmp_left_input = left_test[i:i+batch_size]
            left_infer_result = model(tmp_left_input.to(device))
            left_infer_result_list.append(left_infer_result)

        left_infer_result_list = torch.stack(left_infer_result_list, dim=0).view(-1, 512)

    right_test = list()
    for right_test_path in right_test_paths:
        img = Image.open("/content/inha_data/test/" + right_test_path + '.jpg').convert("RGB") 
        img = data_transform(img)
        right_test.append(img)
    right_test = torch.stack(right_test)

    right_infer_result_list = list()
    with torch.no_grad():
        batch_size = args.bs
        for i in tqdm(range(0, 6000//args.bs)):
            i = i * batch_size
            tmp_right_input = right_test[i:i+batch_size]
            right_infer_result = model(tmp_right_input.to(device))
            right_infer_result_list.append(right_infer_result)

        right_infer_result_list = torch.stack(right_infer_result_list, dim=0).view(-1, 512)


    cosin_similarity = cos_sim(left_infer_result_list, right_infer_result_list)
    submission['answer'] = cosin_similarity.tolist()
    
    submission.to_csv(save_to, index=False)


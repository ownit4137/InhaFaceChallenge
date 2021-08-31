import pandas as pd
from torchvision import transforms
from PIL import Image
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse
from sklearn.metrics import roc_auc_score

import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
sys.path.append('/content/drive/MyDrive/FACE_AI_share/model')
import resnet

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
    parser.add_argument('--size',           type=int,   default = 6000)
    parser.add_argument('--src', type=str, default = 'new.csv')
    parser.add_argument('--img', type=str, default = '/content/inha_data/')
    parser.add_argument('--model', type=str, default = 'resnet18')
    args = parser.parse_args()

    
    load_csv_from = args.src
    load_model_from = './weights/setup/' + args.trained_model + '.pth'
    print('load validation csv file from', load_csv_from)
    print('load model from', load_model_from, '\n')

    submission = pd.read_csv(load_csv_from)

    if args.model == 'resnet18':
      model = resnet18()
    elif args.model == 'resnet50':
      model = resnet50()
    elif args.model == 'resnet101':
      model = resnet101()
    else:
      print('model is not selected')
    
    model.load_state_dict(torch.load(load_model_from, map_location=torch.device('cpu')))
    # model.load_state_dict(torch.load(load_model_from))
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
        img = Image.open(args.img + left_test_path + '.jpg').convert("RGB")
        img = data_transform(img)
        left_test.append(img) 
    left_test = torch.stack(left_test)

    left_infer_result_list = list()
    with torch.no_grad():
        batch_size = args.bs
        for i in range(0, args.size//args.bs):
            i = i * batch_size
            tmp_left_input = left_test[i:i+batch_size]
            left_infer_result = model(tmp_left_input.to(device))
            left_infer_result_list.append(left_infer_result)

        left_infer_result_list = torch.stack(left_infer_result_list, dim=0).view(-1, 512)

    right_test = list()
    for right_test_path in right_test_paths:
        img = Image.open(args.img + right_test_path + '.jpg').convert("RGB") 
        img = data_transform(img)
        right_test.append(img)
    right_test = torch.stack(right_test)

    right_infer_result_list = list()
    with torch.no_grad():
        batch_size = args.bs
        for i in range(0, args.size//args.bs):
            i = i * batch_size
            tmp_right_input = right_test[i:i+batch_size]
            right_infer_result = model(tmp_right_input.to(device))
            right_infer_result_list.append(right_infer_result)

        right_infer_result_list = torch.stack(right_infer_result_list, dim=0).view(-1, 512)


    cosin_similarity = cos_sim(left_infer_result_list, right_infer_result_list)
    submission['pred'] = cosin_similarity.tolist()
    # submission.to_csv('./train_eval/' + args.trained_model + '.csv')

    print(roc_auc_score(submission['answer'], submission['pred']))
    # threshold = [0.935, 0.94, 0.945, 0.95, 0.955, 0.96, 0.965]

    # for t in threshold:
    #   temp = pd.DataFrame()
    #   temp['answer'] = submission['answer'].copy()
    #   temp['pred'] = submission['pred'].copy()
    #   temp['pred'] = temp['pred'].apply(lambda x: 1 if float(x) > t else 0)

    #   accuracy = (temp['pred'] == temp['answer']).sum() / float(len(temp['answer']))
    #   print('threshold: {} | accuracy: {:.4f}'.format(t, accuracy))
import pandas as pd
import bcolz
import os
import numpy as np
from PIL import Image
import argparse
import cv2

def get_val_data(path, name):
    img = bcolz.carray(rootdir = os.path.join(path, name), mode = 'r')
    img_issame = np.load('{}/{}_list.npy'.format(path, name))
    img = img[:, [2, 1, 0], :, :]
    img_left = img[0::2]
    img_right = img[1::2]
    return img_left, img_right, img_issame.astype(int)

def save_val_data(left, right, issame, root):
    df = pd.DataFrame()
    filename = []
    answer = []
    for i, same in enumerate(issame):
        left_img = cv2.normalize(np.moveaxis(left[i], 0, 2), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype('uint8')
        right_img = cv2.normalize(np.moveaxis(right[i], 0, 2), None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype('uint8')
        left_img = Image.fromarray(left_img)
        right_img = Image.fromarray(right_img)
        left_img.save(root + '/valid/left_face_' + str(i) + '.jpg')
        right_img.save(root + '/valid/right_face_' + str(i) + '.jpg')
        filename.append('left_face_' + str(i) + ' right_face_' + str(i)) 
        answer.append(issame[i])
    
    df['face_images'] = filename
    df['answer'] = answer
    df.to_csv(root + '/answer.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',  type=str)
    parser.add_argument('--name',  type=str)
    args = parser.parse_args()

    if not os.path.exists('/content/valid_' + args.name + '/valid'):
        os.mkdir('/content/valid_' + args.name)
        os.mkdir('/content/valid_' + args.name + '/valid')

    left, right, issame = get_val_data(args.root, args.name)
    save_val_data(left, right, issame, '/content/valid_' + args.name)
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os
data_transforms = transforms.Compose([
                        transforms.RandomHorizontalFlip(),
                        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class MS1MDataset(Dataset):
    def __init__(self, eff=False):
        
        root = '/content/inha_data'
        self.file_list = os.path.join(root, 'ID_List.txt')
        
        self.images = []
        self.labels = []
        
        if eff==False:
            self.transformer = data_transforms
        else:
            self.transformer = data_transforms_for_eff
        with open(self.file_list) as f:
            files = f.read().splitlines()
        for i, fi in enumerate(files):
            fi = fi.split()
            image = fi[1] 
            label = int(fi[0])
            self.images.append(os.path.join(root, image))
            self.labels.append(label)
            
    def __getitem__(self, index):

        img = Image.open(self.images[index])
        img = self.transformer(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.images)
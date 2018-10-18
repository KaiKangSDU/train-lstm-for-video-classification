'''
自定义图像导入函数。
与pytorch的数据导入相对接

https://blog.csdn.net/u014380165/article/details/78634829
file resembles  class datasets.ImageFolder


my dataloader, importantly, labels is 0, 1,2,3,4.....rather sad, surprise......
str cannot be converted into tensor.
so, that is why dataloader function can not convert tuple into tensor.

up to now, this file equals to dataloader_default.
'''

from torch.utils.data import  Dataset
from PIL import Image
import os
from torchvision import transforms
import torch


from torchvision import models,transforms
import torch
from torch.autograd import Variable
import torch.nn as nn
import os
import time
import torch.optim as optim
from torch.optim import lr_scheduler





#使用PIL读入图片

def default_loader(path):
    try:
        img = Image.open(path)
        return img.convert('RGB')
    except:
        print("Cannot read image: {}".format(path))

def find_classes(dir):
    classes = [d for d in labels if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

class customData(Dataset):

    #初始化函数主要是要获得图像路径和label， 并且处理成list的形式。
    #并且将图片的预处理操作和提取图片的函数定义好
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None,loader = default_loader):

        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.image_name = [os.path.join(img_path, line.strip()) for line in lines]    #获得图片的地址, 他们的path和label是有默认顺序的。
            #line = Angry\000046280\001.png
            self.img_label = [line.strip().split('\\')[0] for line in lines]              #获得图片的label。 需要把这个path和label处理成list
        classes = []
        for label in self.img_label:
            if label not in classes:
                classes.append(label)
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.data_transforms = data_transforms
        self.loader = loader
        self.dataset = dataset
        self.class_to_idx = class_to_idx
        self.classes = classes

    def __len__(self):
        return len(self.image_name)

    #在这里要把图片的路径和label对应起来
    #主要是读取图片以及将图片和label对应起来。
    #执行预处理操作
    def __getitem__(self, item):
        image_name = self.image_name[item]
        label = self.img_label[item]
        label = self.class_to_idx[label]
        img = self.loader(image_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(image_name))
        return img, label, image_name







if __name__ == '__main__':
    data_transforms = {'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    }


    data_path = "G:\kk_file\EmotiW\AFEW_IMAGE_align_crop\Data" #................................................是要改的
    use_gpu = torch.cuda.is_available()
    batch_size = 32
    num_class = 7

    #convert image into List
    image_datasets = {x: customData(img_path=os.path.join(data_path,x),
                                    txt_path=os.path.join(data_path,x)+'\content.txt',
                                    data_transforms=data_transforms,
                                   dataset=x) for x in ['train', 'val']}




    #convert to tensor, 作为模型可以接受的数据，就定义好是不是打乱， 每个batch_size是多少
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=4)
                   for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    for data in dataloaders['train']:
        inputs, labels, path = data
        print(inputs.shape, labels, path)
        inputs, labels = Variable(inputs), Variable(labels)

















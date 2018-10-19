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
from torchvision import models,transforms
import torch
import os
import glob
import numpy as np
from torch.autograd import Variable
from glob import glob
import skimage.io as io
from skimage.transform import resize
import random
import scipy.io as scio
import torch.nn as nn



#读入向量，整理成LSTM可以输入的形式。

def default_loader(path):
    try:
        clip = glob(path)
            # 按顺序把图片给排好
            # 对视频每一帧的大小进行裁剪，并转化为np.array
        if len(clip) >=16:
            clip = random.sample(clip,16)

        else:
            vi_clip=[]
            print(path)
            for i in range(16):
                vi_clip.append(random.choice(clip))
            clip = vi_clip

        clip = sorted(clip)   #clip是16个mat地址组成的。
        vector = np.array([scio.loadmat(frame)['feature'] for frame in clip])  #(16,1,4096)------->(16,4096)

        vector = np.squeeze(vector)
        assert vector.shape == (16,4096)
        if vector.shape[0] != 16:
            print("The length of the squence is not available")
        if vector.shape[1] != 4096:
            print("The demension of the vector is not available")

        clip = np.float32(vector)

        return torch.from_numpy(clip)
    except:
        print("Cannot read mat file: {}".format(path))
        pass

        #返回值是tensor。 一个视频的tensor




class Videoloader(Dataset):

    #初始化函数主要是要获得图像路径和label， 并且处理成list的形式。
    #并且将图片的预处理操作和提取图片的函数定义好
    def __init__(self, video_path, txt_path, dataset = '', data_transforms=None,loader = default_loader):

        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.video_name = [os.path.join(video_path, line.strip(),'*.mat') for line in lines]    #视频地址= roger/*png
            #line = Angry\000046280
            self.video_label = [line.strip().split('\\')[0] for line in lines]              #获得视频的label。 需要把这个path和label处理成list


        #convert label str into num
        classes = []
        for label in self.video_label:
            if label not in classes:
                classes.append(label)
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        self.data_transforms = data_transforms
        self.loader = loader
        self.dataset = dataset
        self.class_to_idx = class_to_idx
        self.classes = classes

    def __len__(self):
        return len(self.video_name)

    #在这里要把图片的路径和label对应起来
    #主要是读取图片以及将图片和label对应起来。
    #执行预处理操作
    def __getitem__(self, item):
        video_name = self.video_name[item]
        label = self.video_label[item]       #获取标签
        label = self.class_to_idx[label]     #获取数字

        #获取视频
        video = self.loader(video_name)
        #提取视频    video_name= G:\kk_file\EmotiW\AFEW_IMAGE\Data\val\Angry\000149120

        #print(video.shape())
        return video, label

class Rnn(nn.Module):
    def __init__(self, in_dim=4096, hidden_dim=2048, n_class=7,n_layer=1):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.n_class = n_class
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim,n_layer,batch_first=True)
        self.classifier = nn.Linear(hidden_dim,n_class)

    def forward(self,x):
        out,_ = self.lstm(x)
        out = out[:,-1,:]
        out = self.classifier(out)
        return out






if __name__ == '__main__':
    data_transforms = {'train': transforms.Compose([transforms.ToTensor()]),
                        'val': transforms.Compose([ transforms.ToTensor()]),
    }

    data_path = "G:\kk_file\EmotiW\AFEW_IMAGE_align_crop_feature\Data" #................................................是要改的
    use_gpu = torch.cuda.is_available()
    batch_size = 1
    num_class = 7

    #convert image into List. 读入image 和 label, 并且把他们对应起来

    #.................................只有train。。。。。。。。。。。。。。。。。注意！！！！！！！！！！！！！！！！
    #  没有val
    video_datasets = {x: Videoloader(video_path=os.path.join(data_path,x),
                                    txt_path=os.path.join(data_path,x)+'\\video.txt',
                                    data_transforms=data_transforms,
                                   dataset=x) for x in ['train']}

    #convert to tensor, 作为模型可以接受的数据，就定义好是不是打乱， 每个batch_size是多少
    dataloaders = {x: torch.utils.data.DataLoader(video_datasets[x],
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4)
                   for x in ['train']}

    dataset_sizes = {x: len(video_datasets[x]) for x in ['train']}

    model = Rnn()

    for data in dataloaders['train']:
        inputs, labels = data
        print(inputs.shape)
        inputs, labels = Variable(inputs), Variable(labels)   #here, input.shape = batch, seq, input


















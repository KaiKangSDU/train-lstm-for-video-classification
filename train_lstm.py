
'''
提取数据，初始化各种优化器、模型，
放入训练函数汇总进行训练。
主函数
'''

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from torchvision import  models, transforms
from load_for_lstm import Rnn, Videoloader
import time
import os
from torch.utils.data import Dataset
from train import train_model




if __name__ == '__main__':
    data_transforms = {'train': transforms.Compose([transforms.ToTensor()]),
                       'val': transforms.Compose([transforms.ToTensor()]),
                       }

    data_path = "G:\kk_file\EmotiW\AFEW_IMAGE_align_crop_feature\Data"  # ................................................是要改的
    use_gpu = torch.cuda.is_available()
    batch_size = 1
    num_class = 7

    # convert image into List. 读入image 和 label, 并且把他们对应起来

    # .................................只有train。。。。。。。。。。。。。。。。。注意！！！！！！！！！！！！！！！！
    #  没有val
    video_datasets = {x: Videoloader(video_path=os.path.join(data_path, x),
                                     txt_path=os.path.join(data_path, x) + '\\video.txt',
                                     data_transforms=data_transforms,
                                     dataset=x) for x in ['train']}

    # convert to tensor, 作为模型可以接受的数据，就定义好是不是打乱， 每个batch_size是多少
    dataloaders = {x: torch.utils.data.DataLoader(video_datasets[x],
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=4)
                   for x in ['train']}

    dataset_sizes = {x: len(video_datasets[x]) for x in ['train']}

    # get model and replace the original fc layer with your fc layer
    model_ft = Rnn()
    # if use gpu
    if use_gpu:
        model_ft = model_ft.cuda()

    # define cost function
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.005, momentum=0.9)

    # Decay LR by a factor of 0.2 every 5 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.2)

    # multi-GPU
    model_ft = torch.nn.DataParallel(model_ft, device_ids=[0,1])

    # train model
    model_ft = train_model(model=model_ft,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           num_epoches=25,
                           use_gpu=use_gpu,
                           dataloaders = dataloaders,
                           batch_size=batch_size,
                           dataset_sizes=dataset_sizes)

    # save best model
    torch.save(model_ft,"output/best_lstm.pkl")